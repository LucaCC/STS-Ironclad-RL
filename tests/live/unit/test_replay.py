from __future__ import annotations

import json

from sts_ironclad_rl.integration import ActionCommand
from sts_ironclad_rl.live import (
    REPLAY_SCHEMA_VERSION,
    ActionDecision,
    JsonlReplaySink,
    ReplayEntry,
    replay_entry_to_dict,
)
from tests.live.factories import encode_snapshot, make_combat_snapshot


def test_replay_entry_to_dict_emits_stable_schema() -> None:
    snapshot = make_combat_snapshot()
    observation = encode_snapshot(snapshot)
    replay_payload = replay_entry_to_dict(
        ReplayEntry(
            session_id=snapshot.session_id,
            step_index=0,
            observation=observation,
            action=ActionDecision(action_id="end_turn"),
            command=ActionCommand(session_id=snapshot.session_id, command="end"),
            reward=1.0,
            metadata={"source": "test"},
        )
    )

    assert replay_payload["schema_version"] == REPLAY_SCHEMA_VERSION
    assert replay_payload["observation"]["snapshot"]["session_id"] == snapshot.session_id
    assert replay_payload["action"]["action_id"] == "end_turn"
    assert replay_payload["command"]["command"] == "end"


def test_jsonl_replay_sink_appends_serialized_entries(tmp_path) -> None:
    snapshot = make_combat_snapshot()
    observation = encode_snapshot(snapshot)
    sink = JsonlReplaySink(output_path=tmp_path / "replay.jsonl")

    sink.log(
        ReplayEntry(
            session_id=snapshot.session_id,
            step_index=1,
            observation=observation,
            terminal=True,
            outcome="victory",
        )
    )

    payload = json.loads(sink.output_path.read_text(encoding="utf-8").strip())
    assert payload["schema_version"] == REPLAY_SCHEMA_VERSION
    assert payload["observation"]["snapshot"]["session_id"] == observation.snapshot.session_id
    assert payload["terminal"] is True
    assert payload["outcome"] == "victory"
