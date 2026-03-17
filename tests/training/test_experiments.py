from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from sts_ironclad_rl.integration import GameStateSnapshot
from sts_ironclad_rl.live import (
    ActionDecision,
    EncodedObservation,
    ReplayEntry,
    RolloutResult,
)
from sts_ironclad_rl.training import ExperimentArtifactStore, ExperimentRunner, ExperimentSpec


@dataclass
class StubRolloutRunner:
    results: tuple[RolloutResult, ...]
    calls: list[tuple[str, int | None]] = field(default_factory=list)
    _index: int = 0

    def run_episode(self, *, policy, evaluation_case=None) -> RolloutResult:
        self.calls.append((policy.name, evaluation_case.max_steps if evaluation_case else None))
        result = self.results[self._index]
        self._index += 1
        return result


@dataclass(frozen=True)
class StubPolicy:
    name: str = "simple_heuristic"

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        del observation
        return ActionDecision(action_id="end_turn")


def test_experiment_runner_collects_with_shared_rollout_runner(tmp_path) -> None:
    runner = StubRolloutRunner(results=(_result("session-1"), _result("session-2")))
    experiment_runner = ExperimentRunner(
        rollout_runner=runner,
        artifact_store=ExperimentArtifactStore(root_dir=tmp_path),
        clock=_fixed_clock(
            datetime(2026, 3, 17, 18, 30, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 18, 31, tzinfo=timezone.utc),
        ),
    )
    spec = ExperimentSpec(
        experiment_name="heuristic collection",
        policy_name="simple_heuristic",
        episode_count=2,
        evaluation_case_name="collection",
        max_steps=15,
    )

    result = experiment_runner.run(spec=spec, policy=StubPolicy())

    assert len(result.episodes) == 2
    assert runner.calls == [("simple_heuristic", 15), ("simple_heuristic", 15)]
    assert result.metadata.started_at == "2026-03-17T18:30:00Z"
    assert result.metadata.completed_at == "2026-03-17T18:31:00Z"
    assert result.output_dir.joinpath("trajectory.jsonl").exists()


def test_experiment_runner_rejects_policy_name_mismatch(tmp_path) -> None:
    experiment_runner = ExperimentRunner(
        rollout_runner=StubRolloutRunner(results=(_result("session-1"),)),
        artifact_store=ExperimentArtifactStore(root_dir=tmp_path),
    )
    spec = ExperimentSpec(
        experiment_name="bad config",
        policy_name="random_legal",
        episode_count=1,
    )

    with pytest.raises(ValueError, match="policy_name"):
        experiment_runner.run(spec=spec, policy=StubPolicy())


def _result(session_id: str) -> RolloutResult:
    snapshot = GameStateSnapshot(
        session_id=session_id,
        screen_state="COMBAT",
        available_actions=("end_turn",),
        in_combat=True,
        floor=3,
        act=1,
        raw_state={"reward": 1.0},
    )
    observation = EncodedObservation(
        snapshot=snapshot,
        legal_action_ids=snapshot.available_actions,
        features={"turn": 1},
        metadata={},
    )
    return RolloutResult(
        session_id=session_id,
        entries=(
            ReplayEntry(
                session_id=session_id,
                step_index=0,
                observation=observation,
                action=ActionDecision(action_id="end_turn"),
                reward=1.0,
            ),
        ),
        terminal=True,
        step_count=1,
        outcome="combat_end",
        total_reward=1.0,
        metadata={"final_floor": 3},
    )


def _fixed_clock(*values: datetime):
    iterator = iter(values)

    def _next() -> datetime:
        return next(iterator)

    return _next
