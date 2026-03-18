from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from sts_ironclad_rl.integration import ActionCommand, GameStateSnapshot
from sts_ironclad_rl.live import (
    ActionDecision,
    EncodedObservation,
    ReplayEntry,
    RolloutResult,
    summarize_rollouts,
)
from sts_ironclad_rl.training import (
    ExperimentArtifactStore,
    ExperimentSpec,
    create_dqn_trainer_run_layout,
    make_run_metadata,
    resolve_dqn_training_summary_path,
)


def test_artifact_store_uses_stable_run_layout_and_writes_outputs(tmp_path) -> None:
    store = ExperimentArtifactStore(root_dir=tmp_path)
    spec = ExperimentSpec(
        experiment_name="Random Legal Smoke",
        policy_name="random_legal",
        episode_count=1,
        evaluation_case_name="smoke",
    )
    started_at = datetime(2026, 3, 17, 18, 0, tzinfo=timezone.utc)
    completed_at = datetime(2026, 3, 17, 18, 0, 3, tzinfo=timezone.utc)
    layout = store.create_run_layout(spec=spec, started_at=started_at)
    episode = _episode_result()
    summary = summarize_rollouts(
        results=(episode,),
        policy_name=spec.policy_name,
        case_name=spec.evaluation_case_name,
    )

    store.write_run_artifacts(
        layout=layout,
        spec=spec,
        metadata=make_run_metadata(
            layout=layout,
            spec=spec,
            started_at=started_at,
            completed_at=completed_at,
        ),
        summary=summary,
        episodes=(episode,),
    )

    assert layout.run_dir.relative_to(tmp_path).parts[0] == "random-legal-smoke"
    assert layout.run_dir.name == f"20260317T180000Z-{spec.fingerprint()[:8]}"
    config_payload = json.loads(layout.config_path.read_text(encoding="utf-8"))
    assert config_payload["policy_name"] == "random_legal"
    assert json.loads(layout.summary_path.read_text(encoding="utf-8"))["outcome_counts"] == {
        "victory": 1
    }

    episode_lines = layout.episodes_path.read_text(encoding="utf-8").strip().splitlines()
    trajectory_lines = layout.trajectory_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(episode_lines) == 1
    assert len(trajectory_lines) == 2
    assert json.loads(trajectory_lines[0])["entry"]["schema_version"] == "live_replay.v1"
    assert json.loads(trajectory_lines[0])["entry"]["action"]["action_id"] == "play_0"


def test_dqn_trainer_artifact_layout_and_summary_resolution_are_canonical(tmp_path) -> None:
    layout = create_dqn_trainer_run_layout(tmp_path / "masked_dqn_baseline")

    assert layout.config_path == layout.root_dir / "config.json"
    assert layout.summary_path == layout.root_dir / "summary.json"
    assert layout.final_checkpoint_path == layout.checkpoints_dir / "checkpoint_final.pt"
    assert resolve_dqn_training_summary_path(layout.final_checkpoint_path) == layout.summary_path


def test_resolve_dqn_training_summary_path_rejects_noncanonical_checkpoint_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="checkpoints/"):
        resolve_dqn_training_summary_path(tmp_path / "checkpoint_final.pt")


def _episode_result() -> RolloutResult:
    snapshot = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("play_0", "end_turn"),
        in_combat=True,
        floor=3,
        act=1,
        raw_state={"turn": 1},
    )
    observation = EncodedObservation(
        snapshot=snapshot,
        legal_action_ids=snapshot.available_actions,
        features={"turn": 1},
        metadata={"screen_state": snapshot.screen_state},
    )
    entries = (
        ReplayEntry(
            session_id=snapshot.session_id,
            step_index=0,
            observation=observation,
            action=ActionDecision(action_id="play_0"),
            command=ActionCommand(session_id=snapshot.session_id, command="play", arguments={}),
        ),
        ReplayEntry(
            session_id=snapshot.session_id,
            step_index=1,
            observation=observation,
            terminal=True,
            outcome="victory",
        ),
    )
    return RolloutResult(
        session_id=snapshot.session_id,
        entries=entries,
        terminal=True,
        step_count=1,
        outcome="victory",
        total_reward=2.0,
        metadata={"final_floor": 3},
    )
