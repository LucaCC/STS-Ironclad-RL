from __future__ import annotations

import json

from sts_ironclad_rl.integration import (
    ActionCommand,
    GameStateSnapshot,
    JsonlTrajectoryLogger,
    TrajectoryEntry,
)


def test_jsonl_trajectory_logger_writes_one_entry_per_line(tmp_path) -> None:
    output_path = tmp_path / "logs" / "trajectory.jsonl"
    logger = JsonlTrajectoryLogger(output_path)
    entry = TrajectoryEntry(
        session_id="session-1",
        step_index=0,
        observation=GameStateSnapshot(
            session_id="session-1",
            screen_state="COMBAT",
            available_actions=("end_turn",),
            in_combat=True,
            raw_state={"player_hp": 80},
        ),
        action=ActionCommand(session_id="session-1", command="end_turn"),
        reward=None,
        terminal=False,
        metadata={"source": "smoke-test"},
    )

    logger.log(entry)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "action": {
            "arguments": {},
            "command": "end_turn",
            "session_id": "session-1",
        },
        "metadata": {"source": "smoke-test"},
        "observation": {
            "act": None,
            "available_actions": ["end_turn"],
            "floor": None,
            "in_combat": True,
            "raw_state": {"player_hp": 80},
            "screen_state": "COMBAT",
            "session_id": "session-1",
        },
        "reward": None,
        "session_id": "session-1",
        "step_index": 0,
        "terminal": False,
    }
