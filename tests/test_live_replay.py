from __future__ import annotations

import json

import pytest

from sts_ironclad_rl.integration import ActionCommand, GameStateSnapshot
from sts_ironclad_rl.live import (
    REPLAY_SCHEMA_VERSION,
    ActionDecision,
    JsonlReplayWriter,
    RawStateObservationEncoder,
    ReplayEntry,
    ReplayFailure,
    read_replay_entries,
)


def _observation():
    snapshot = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("play", "end"),
        in_combat=True,
        floor=3,
        act=1,
        raw_state={
            "combat_state": {
                "turn": 2,
                "player": {"current_hp": 72, "max_hp": 80, "energy": 1},
                "hand": [{"is_playable": True, "has_target": False, "cost": 1}],
                "monsters": [{"current_hp": 20, "is_gone": False}],
            }
        },
    )
    return RawStateObservationEncoder().encode(snapshot)


def test_jsonl_replay_round_trip_preserves_structured_fields(tmp_path) -> None:
    output_path = tmp_path / "replays" / "episode.jsonl"
    writer = JsonlReplayWriter(output_path)
    entry = ReplayEntry(
        session_id="session-1",
        step_index=2,
        recorded_at="2026-03-17T13:00:00.000+00:00",
        observation=_observation(),
        state_reference={"trace_id": "raw-17", "snapshot_index": 8},
        action=ActionDecision(
            action_id="play_card:0",
            arguments={"target": 0},
            metadata={"policy": "greedy"},
        ),
        command=ActionCommand(
            session_id="session-1",
            command="play",
            arguments={"card_index": 1, "target_index": 0},
        ),
        reward=12.5,
        metadata={"run_id": "run-1"},
    )

    writer.log(entry)

    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["schema_version"] == REPLAY_SCHEMA_VERSION
    assert payload["observation"]["metadata"]["screen_state"] == "COMBAT"
    assert payload["legal_action_ids"] == ["play_card:0", "end_turn"]
    assert payload["command"]["command"] == "play"

    assert read_replay_entries(output_path) == (entry,)


def test_jsonl_replay_captures_terminal_failure_metadata(tmp_path) -> None:
    output_path = tmp_path / "replays" / "failed_episode.jsonl"
    writer = JsonlReplayWriter(output_path)
    failure = ReplayFailure(
        stage="bridge_receive",
        message="timed out waiting for state",
        error_type="TimeoutError",
        recoverable=True,
        details={"timeout_seconds": 5.0},
    )
    entry = ReplayEntry(
        session_id="session-1",
        step_index=5,
        recorded_at="2026-03-17T13:05:00.000+00:00",
        observation=_observation(),
        terminal=True,
        outcome="interrupted",
        failure=failure,
        metadata={"run_status": "aborted"},
    )

    writer.log(entry)

    loaded_entry = read_replay_entries(output_path)[0]
    assert loaded_entry.failure == failure
    assert loaded_entry.outcome == "interrupted"
    assert loaded_entry.terminal is True


def test_jsonl_replay_supports_rollout_style_readback(tmp_path) -> None:
    output_path = tmp_path / "replays" / "episode.jsonl"
    writer = JsonlReplayWriter(output_path)
    entries = (
        ReplayEntry(
            session_id="session-1",
            step_index=0,
            recorded_at="2026-03-17T13:00:00.000+00:00",
            observation=_observation(),
            action=ActionDecision(action_id="play_card:0"),
            command=ActionCommand(
                session_id="session-1",
                command="play",
                arguments={"card_index": 1},
            ),
            reward=6.0,
        ),
        ReplayEntry(
            session_id="session-1",
            step_index=1,
            recorded_at="2026-03-17T13:00:01.000+00:00",
            observation=_observation(),
            action=ActionDecision(action_id="end_turn"),
            command=ActionCommand(session_id="session-1", command="end"),
            reward=0.0,
            terminal=True,
            outcome="victory",
        ),
    )

    writer.write_all(entries)
    loaded_entries = read_replay_entries(output_path)

    assert [entry.step_index for entry in loaded_entries] == [0, 1]
    assert [entry.command.command for entry in loaded_entries if entry.command is not None] == [
        "play",
        "end",
    ]
    assert sum(entry.reward or 0.0 for entry in loaded_entries) == pytest.approx(6.0)
    assert loaded_entries[-1].outcome == "victory"


def test_jsonl_replay_rejects_unknown_schema_version(tmp_path) -> None:
    output_path = tmp_path / "replays" / "bad.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        '{"schema_version":"unknown","session_id":"session-1"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsupported replay schema version"):
        read_replay_entries(output_path)
