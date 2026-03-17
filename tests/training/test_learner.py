from __future__ import annotations

from typing import Any

import pytest

from sts_ironclad_rl.integration import GameStateSnapshot
from sts_ironclad_rl.live import ActionDecision, BridgeObservationEncoder, ReplayEntry
from sts_ironclad_rl.training import (
    LEARNER_OBSERVATION_SCHEMA_VERSION,
    LEARNER_TRANSITION_SCHEMA_VERSION,
    LearnerActionIndex,
    LearnerObservationEncoder,
    LearnerRewardFunction,
    LearnerTransitionExtractor,
    RewardConfig,
)


def make_card(
    *,
    name: str,
    card_id: str | None = None,
    cost: int = 1,
    is_playable: bool = True,
    has_target: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "id": card_id or name.lower().replace(" ", "_"),
        "cost": cost,
        "is_playable": is_playable,
        "has_target": has_target,
    }


def make_enemy(
    *,
    name: str = "Louse",
    current_hp: int = 18,
    max_hp: int | None = None,
    block: int = 0,
    intent: str | None = None,
    is_gone: bool = False,
    half_dead: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "id": name.lower().replace(" ", "_"),
        "current_hp": current_hp,
        "max_hp": current_hp if max_hp is None else max_hp,
        "block": block,
        "intent": intent,
        "intent_base_damage": 0,
        "intent_hits": 1,
        "is_gone": is_gone,
        "half_dead": half_dead,
    }


def make_combat_state(
    *,
    turn: int = 1,
    player: dict[str, Any] | None = None,
    hand: tuple[dict[str, Any], ...] = (),
    monsters: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    return {
        "turn": turn,
        "player": {
            "current_hp": 70,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
        }
        if player is None
        else dict(player),
        "hand": [dict(card) for card in hand],
        "monsters": [dict(monster) for monster in monsters],
    }


def make_snapshot(
    *,
    session_id: str = "session-1",
    screen_state: str = "COMBAT",
    available_actions: tuple[str, ...] = ("play", "end"),
    in_combat: bool = True,
    raw_state: dict[str, Any] | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        session_id=session_id,
        screen_state=screen_state,
        available_actions=available_actions,
        in_combat=in_combat,
        floor=3,
        act=1,
        raw_state={} if raw_state is None else dict(raw_state),
    )


def make_combat_snapshot(
    *,
    turn: int = 1,
    player: dict[str, Any] | None = None,
    hand: tuple[dict[str, Any], ...] = (),
    monsters: tuple[dict[str, Any], ...] = (),
    available_actions: tuple[str, ...] = ("play", "end"),
) -> GameStateSnapshot:
    return make_snapshot(
        available_actions=available_actions,
        in_combat=True,
        raw_state={
            "combat_state": make_combat_state(
                turn=turn,
                player=player,
                hand=hand,
                monsters=monsters,
            )
        },
    )


def encode_snapshot(snapshot: GameStateSnapshot):
    return BridgeObservationEncoder().encode(snapshot)


def test_learner_observation_encoder_emits_fixed_size_vector_with_padding_masks() -> None:
    snapshot = make_snapshot(
        available_actions=("play", "end"),
        in_combat=True,
        raw_state={
            "combat_state": {
                **make_combat_state(
                    turn=7,
                    player={"current_hp": 41, "max_hp": 80, "block": 12, "energy": 2},
                    hand=(
                        make_card(name="Bash", card_id="bash", cost=2, has_target=True),
                        make_card(name="Defend", card_id="defend", cost=1, has_target=False),
                    ),
                    monsters=(
                        make_enemy(name="Jaw Worm", current_hp=34, block=6, intent="ATTACK"),
                        make_enemy(
                            name="Cultist",
                            current_hp=0,
                            block=0,
                            intent="BUFF",
                            is_gone=False,
                            half_dead=False,
                        ),
                    ),
                ),
                "draw_pile": [{}, {}, {}],
                "discard_pile": [{}],
                "exhaust_pile": [{}, {}],
            }
        },
    )

    encoder = LearnerObservationEncoder()
    encoded = encoder.encode(snapshot)
    feature_names = encoder.layout.feature_names()

    assert encoded.schema_version == LEARNER_OBSERVATION_SCHEMA_VERSION
    assert len(encoded.vector) == encoder.layout.vector_size == 93
    assert len(feature_names) == 93
    assert encoded.hand_mask == (1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    assert encoded.enemy_mask == (1, 1, 0, 0, 0)

    features = dict(zip(feature_names, encoded.vector, strict=True))
    assert features["player_hp"] == 41.0
    assert features["player_max_hp"] == 80.0
    assert features["player_block"] == 12.0
    assert features["player_energy"] == 2.0
    assert features["draw_pile_size"] == 3.0
    assert features["discard_pile_size"] == 1.0
    assert features["exhaust_pile_size"] == 2.0
    assert features["turn_index"] == 7.0
    assert features["hand_0_cost"] == 2.0
    assert features["hand_0_has_target"] == 1.0
    assert features["hand_1_is_playable"] == 1.0
    assert features["hand_2_card_token"] == 0.0
    assert features["enemy_0_current_hp"] == 34.0
    assert features["enemy_0_alive"] == 1.0
    assert features["enemy_1_current_hp"] == 0.0
    assert features["enemy_1_alive"] == 0.0
    assert features["enemy_2_intent_token"] == 0.0


def test_learner_action_index_has_stable_semantics() -> None:
    action_index = LearnerActionIndex()

    assert action_index.size == 61
    assert action_index.describe(0) == "END_TURN"
    assert action_index.describe(1) == "PLAY_CARD_HAND_0_NO_TARGET"
    assert action_index.describe(10) == "PLAY_CARD_HAND_9_NO_TARGET"
    assert action_index.describe(11) == "PLAY_CARD_HAND_0_TARGET_ENEMY_0"
    assert action_index.describe(15) == "PLAY_CARD_HAND_0_TARGET_ENEMY_4"
    assert action_index.describe(16) == "PLAY_CARD_HAND_1_TARGET_ENEMY_0"
    assert action_index.action_to_index("end_turn") == 0
    assert action_index.action_to_index("play_card:3") == 4
    assert action_index.action_to_index("play_card:3:4") == 30
    assert action_index.index_to_action_id(0) == "end_turn"
    assert action_index.index_to_action_id(4) == "play_card:3"
    assert action_index.index_to_action_id(30) == "play_card:3:4"


def test_learner_action_mask_matches_index_order_and_filters_unrepresentable_actions() -> None:
    snapshot = make_combat_snapshot(
        hand=(
            make_card(name="Strike", has_target=True, is_playable=True),
            make_card(name="Burn", has_target=False, is_playable=False),
            make_card(name="Defend", has_target=False, is_playable=True),
        ),
        monsters=(
            make_enemy(name="Jaw Worm", current_hp=24),
            make_enemy(name="Cultist", current_hp=0),
        ),
    )

    action_index = LearnerActionIndex()
    mask = action_index.legal_mask(snapshot)

    assert len(mask) == action_index.size
    assert mask[0] == 1
    assert mask[action_index.action_to_index("play_card:0")] == 0
    assert mask[action_index.action_to_index("play_card:0:0")] == 1
    assert mask[action_index.action_to_index("play_card:1")] == 0
    assert mask[action_index.action_to_index("play_card:2")] == 1


def test_learner_reward_function_applies_hp_deltas_and_terminal_bonuses() -> None:
    reward_function = LearnerRewardFunction(
        config=RewardConfig(
            enemy_hp_weight=0.1,
            player_hp_weight=0.2,
            terminal_win_bonus=1.0,
            terminal_loss_penalty=2.0,
            step_penalty=0.01,
        )
    )
    current = make_combat_snapshot(
        player={"current_hp": 50, "max_hp": 80, "block": 0, "energy": 3},
        monsters=(make_enemy(name="Jaw Worm", current_hp=20),),
    )
    nxt = make_combat_snapshot(
        player={"current_hp": 47, "max_hp": 80, "block": 0, "energy": 2},
        monsters=(make_enemy(name="Jaw Worm", current_hp=12),),
    )
    terminal_victory = make_snapshot(
        screen_state="MAP",
        available_actions=("proceed",),
        in_combat=False,
        raw_state={"victory": True},
    )
    terminal_defeat = make_snapshot(
        screen_state="DEATH",
        available_actions=(),
        in_combat=False,
        raw_state={"player_dead": True},
    )

    assert reward_function.reward(current, nxt, done=False, outcome=None) == pytest.approx(0.19)
    assert reward_function.reward(
        nxt, terminal_victory, done=True, outcome="victory"
    ) == pytest.approx(2.19)
    assert reward_function.reward(
        nxt, terminal_defeat, done=True, outcome="defeat"
    ) == pytest.approx(-11.41)


def test_transition_extractor_builds_replay_compatible_learning_tuples() -> None:
    extractor = LearnerTransitionExtractor()
    first_snapshot = make_combat_snapshot(
        hand=(make_card(name="Strike", has_target=True),),
        monsters=(make_enemy(name="Jaw Worm", current_hp=20),),
    )
    second_snapshot = make_combat_snapshot(
        turn=2,
        player={"current_hp": 77, "max_hp": 80, "block": 0, "energy": 1},
        hand=(make_card(name="Defend", has_target=False),),
        monsters=(make_enemy(name="Jaw Worm", current_hp=14),),
    )
    terminal_snapshot = make_snapshot(
        screen_state="MAP",
        available_actions=("proceed",),
        in_combat=False,
        raw_state={"victory": True},
    )
    entries = (
        ReplayEntry(
            session_id="session-1",
            step_index=0,
            observation=encode_snapshot(first_snapshot),
            action=ActionDecision(action_id="play_card:0:0"),
        ),
        ReplayEntry(
            session_id="session-1",
            step_index=1,
            observation=encode_snapshot(second_snapshot),
            action=ActionDecision(action_id="play_card:0"),
        ),
        ReplayEntry(
            session_id="session-1",
            step_index=2,
            observation=encode_snapshot(terminal_snapshot),
            terminal=True,
            outcome="victory",
        ),
    )

    transitions = extractor.extract(entries)

    assert len(transitions) == 2
    assert all(
        transition.schema_version == LEARNER_TRANSITION_SCHEMA_VERSION for transition in transitions
    )
    assert transitions[0].action_index == 11
    assert transitions[0].done is False
    assert transitions[0].mask[0] == 1
    assert transitions[0].mask[extractor.action_index.action_to_index("play_card:0")] == 1
    assert transitions[1].done is True
    assert sum(transitions[1].mask) == 0
    assert len(transitions[0].state) == extractor.observation_encoder.layout.vector_size
    assert len(transitions[0].next_state) == extractor.observation_encoder.layout.vector_size
    assert transitions[0].as_tuple()[1] == 11


def test_transition_extractor_ignores_terminal_only_entries_without_actions() -> None:
    terminal_snapshot = make_snapshot(
        screen_state="MAP",
        available_actions=("proceed",),
        in_combat=False,
        raw_state={"victory": True},
    )
    entries = (
        ReplayEntry(
            session_id="session-1",
            step_index=0,
            observation=encode_snapshot(terminal_snapshot),
            terminal=True,
            outcome="victory",
        ),
    )

    assert LearnerTransitionExtractor().extract(entries) == ()


def test_learner_mask_truncates_overflowing_hand_and_enemy_slots() -> None:
    snapshot = make_combat_snapshot(
        hand=tuple(make_card(name=f"Card {index}") for index in range(12)),
        monsters=tuple(make_enemy(name=f"Enemy {index}") for index in range(6)),
    )
    encoded = LearnerObservationEncoder().encode(snapshot)

    assert encoded.hand_mask == (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert encoded.enemy_mask == (1, 1, 1, 1, 1)
