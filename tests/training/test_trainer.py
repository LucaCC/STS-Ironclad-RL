from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from sts_ironclad_rl.integration import GameStateSnapshot
from sts_ironclad_rl.live import (
    ActionDecision,
    BridgeObservationEncoder,
    EndTurnAction,
    PlayCardAction,
    ReplayEntry,
    RolloutResult,
)
from sts_ironclad_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    EpsilonSchedule,
    MaskedDQNConfig,
    TrainerState,
    TrainingEpisodeMetrics,
    should_sync_target_network,
    summarize_training_metrics,
)


@dataclass
class StaticRolloutRunner:
    results: list[RolloutResult]
    calls: int = field(init=False, default=0)

    def run_episode(self, *, policy, evaluation_case=None) -> RolloutResult:
        del policy, evaluation_case
        index = min(self.calls, len(self.results) - 1)
        self.calls += 1
        return self.results[index]


def test_epsilon_schedule_decays_linearly_and_clamps() -> None:
    schedule = EpsilonSchedule(initial=1.0, final=0.2, decay_steps=100)

    assert schedule.value(0) == pytest.approx(1.0)
    assert schedule.value(25) == pytest.approx(0.8)
    assert schedule.value(100) == pytest.approx(0.2)
    assert schedule.value(200) == pytest.approx(0.2)


def test_trainer_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="batch_size must not exceed replay_buffer_size"):
        DQNTrainerConfig(replay_buffer_size=4, batch_size=5)
    with pytest.raises(ValueError, match="target_update_frequency must be positive"):
        DQNTrainerConfig(target_update_frequency=0)
    with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
        DQNTrainerConfig(gamma=1.5)


def test_target_sync_cadence_helper() -> None:
    assert should_sync_target_network(optimization_steps=1, target_update_frequency=2) is False
    assert should_sync_target_network(optimization_steps=2, target_update_frequency=2) is True
    assert should_sync_target_network(optimization_steps=4, target_update_frequency=2) is True


def test_trainer_runs_update_loop_with_replay_backed_transitions(tmp_path) -> None:
    trainer = DQNTrainer(
        rollout_runner=StaticRolloutRunner(results=[make_rollout_result()]),
        config=DQNTrainerConfig(
            train_episodes=1,
            evaluation_cadence=10,
            checkpoint_cadence=10,
            evaluation_episodes=1,
            replay_buffer_size=8,
            batch_size=2,
            warmup_steps=0,
            target_update_frequency=1,
            network=MaskedDQNConfig(hidden_sizes=(8,)),
            seed=7,
        ),
    )
    initial_parameters = [
        parameter.detach().clone() for parameter in trainer.online_network.parameters()
    ]

    result = trainer.train(output_dir=tmp_path)

    assert result.state == TrainerState(
        completed_episodes=1,
        environment_steps=2,
        optimization_steps=1,
        target_sync_steps=1,
    )
    assert len(trainer.replay_buffer) == 2
    assert result.episode_metrics[0].episode_return != 0.0
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_final.pt").exists()
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            initial_parameters, trainer.online_network.parameters(), strict=True
        )
    )


def test_trainer_checkpoint_restores_progress_and_replay(tmp_path) -> None:
    config = DQNTrainerConfig(
        train_episodes=1,
        evaluation_cadence=10,
        checkpoint_cadence=10,
        evaluation_episodes=1,
        replay_buffer_size=8,
        batch_size=2,
        warmup_steps=0,
        network=MaskedDQNConfig(hidden_sizes=(8,)),
        seed=11,
    )
    trainer = DQNTrainer(
        rollout_runner=StaticRolloutRunner(results=[make_rollout_result()]),
        config=config,
    )
    trainer.train()
    checkpoint_path = tmp_path / "trainer.pt"
    trainer.save_checkpoint(checkpoint_path, metadata={"label": "resume-test"})

    restored = DQNTrainer(
        rollout_runner=StaticRolloutRunner(results=[make_rollout_result()]),
        config=config,
    )
    metadata = restored.load_checkpoint(checkpoint_path)

    assert metadata["label"] == "resume-test"
    assert metadata["state"] == trainer.training_summary()["state"]
    assert restored.state == trainer.state
    assert len(restored.replay_buffer) == len(trainer.replay_buffer)


def test_training_metric_summary_aggregates_recent_window() -> None:
    summary = summarize_training_metrics(
        (
            TrainingEpisodeMetrics(
                episode_index=0,
                environment_steps=3,
                optimization_steps=1,
                replay_size=3,
                epsilon=0.9,
                transition_count=3,
                episode_return=1.5,
                average_reward=0.5,
                average_loss=0.2,
                episode_length=3,
                outcome="victory",
                terminal=True,
                total_reward_proxy=1.0,
                mask_fallback_count=1,
                invalid_action_count=0,
            ),
            TrainingEpisodeMetrics(
                episode_index=1,
                environment_steps=6,
                optimization_steps=2,
                replay_size=6,
                epsilon=0.8,
                transition_count=3,
                episode_return=0.3,
                average_reward=0.1,
                average_loss=0.4,
                episode_length=4,
                outcome="defeat",
                terminal=True,
                total_reward_proxy=-1.0,
                mask_fallback_count=0,
                invalid_action_count=2,
            ),
        )
    )

    assert summary["episode_count"] == 2
    assert summary["average_loss"] == pytest.approx(0.3)
    assert summary["average_reward"] == pytest.approx(0.3)
    assert summary["average_episode_return"] == pytest.approx(0.9)
    assert summary["average_episode_length"] == pytest.approx(3.5)
    assert summary["outcomes"] == {"defeat": 1, "victory": 1}
    assert summary["mask_fallback_count"] == 1
    assert summary["invalid_action_count"] == 2
    assert summary["epsilon"] == pytest.approx(0.8)


def make_rollout_result() -> RolloutResult:
    play_action = PlayCardAction(hand_index=0).action_id
    end_turn = EndTurnAction().action_id

    first_snapshot = make_combat_snapshot(
        available_actions=(play_action, end_turn),
        hand=(make_card(name="Strike", has_target=False),),
        monsters=(make_enemy(current_hp=18, max_hp=18),),
    )
    second_snapshot = make_combat_snapshot(
        available_actions=(end_turn,),
        turn=2,
        player={"current_hp": 70, "max_hp": 80, "block": 0, "energy": 2},
        hand=(),
        monsters=(make_enemy(current_hp=12, max_hp=18),),
    )
    terminal_snapshot = make_snapshot(
        available_actions=(),
        screen_state="MAP",
        in_combat=False,
        raw_state={"victory": True},
    )

    return RolloutResult(
        session_id="session-1",
        entries=(
            ReplayEntry(
                session_id="session-1",
                step_index=0,
                observation=encode_snapshot(first_snapshot),
                action=ActionDecision(action_id=play_action),
            ),
            ReplayEntry(
                session_id="session-1",
                step_index=1,
                observation=encode_snapshot(second_snapshot),
                action=ActionDecision(action_id=end_turn),
            ),
            ReplayEntry(
                session_id="session-1",
                step_index=2,
                observation=encode_snapshot(terminal_snapshot),
                terminal=True,
                outcome="victory",
            ),
        ),
        terminal=True,
        step_count=2,
        outcome="victory",
        total_reward=1.0,
    )


def make_snapshot(
    *,
    session_id: str = "session-1",
    screen_state: str = "COMBAT",
    available_actions: tuple[str, ...] = (),
    in_combat: bool = True,
    floor: int | None = 3,
    act: int | None = 1,
    raw_state: dict[str, object] | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        session_id=session_id,
        screen_state=screen_state,
        available_actions=available_actions,
        in_combat=in_combat,
        floor=floor,
        act=act,
        raw_state={} if raw_state is None else dict(raw_state),
    )


def make_combat_snapshot(
    *,
    session_id: str = "session-1",
    available_actions: tuple[str, ...] = (),
    turn: int = 1,
    player: dict[str, object] | None = None,
    hand: tuple[dict[str, object], ...] = (),
    monsters: tuple[dict[str, object], ...] = (),
    extra_raw_state: dict[str, object] | None = None,
) -> GameStateSnapshot:
    raw_state: dict[str, object] = {
        "combat_state": {
            "turn": turn,
            "player": player or {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3},
            "hand": [dict(card) for card in hand],
            "monsters": [dict(monster) for monster in monsters],
        }
    }
    if extra_raw_state is not None:
        raw_state.update(extra_raw_state)
    return make_snapshot(
        session_id=session_id,
        available_actions=available_actions,
        in_combat=True,
        raw_state=raw_state,
    )


def make_card(
    *,
    name: str,
    cost: int = 1,
    is_playable: bool = True,
    has_target: bool = False,
) -> dict[str, object]:
    return {
        "name": name,
        "id": name.lower(),
        "cost": cost,
        "is_playable": is_playable,
        "has_target": has_target,
    }


def make_enemy(
    *,
    current_hp: int,
    max_hp: int,
    intent: str | None = None,
) -> dict[str, object]:
    return {
        "name": "Louse",
        "id": "louse",
        "current_hp": current_hp,
        "max_hp": max_hp,
        "block": 0,
        "intent": intent,
        "intent_base_damage": 0,
        "intent_hits": 1,
        "is_gone": False,
        "half_dead": False,
    }


def encode_snapshot(snapshot: GameStateSnapshot):
    return BridgeObservationEncoder().encode(snapshot)
