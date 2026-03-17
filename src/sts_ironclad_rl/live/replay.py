"""Structured replay serialization helpers for the live rollout pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..integration import ActionCommand, GameStateSnapshot
from .contracts import ActionDecision, EncodedObservation, ReplayEntry

REPLAY_SCHEMA_VERSION = "live_replay.v1"


def replay_entry_to_dict(entry: ReplayEntry) -> dict[str, Any]:
    """Return a stable JSON-serializable payload for one replay entry."""

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "session_id": entry.session_id,
        "step_index": entry.step_index,
        "observation": _observation_to_dict(entry.observation),
        "action": _action_to_dict(entry.action),
        "command": _command_to_dict(entry.command),
        "reward": entry.reward,
        "terminal": entry.terminal,
        "outcome": entry.outcome,
        "metadata": dict(entry.metadata),
    }


@dataclass
class JsonlReplaySink:
    """Append structured replay entries to a JSONL file."""

    output_path: Path

    def log(self, entry: ReplayEntry) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(replay_entry_to_dict(entry), sort_keys=True))
            handle.write("\n")


def _observation_to_dict(observation: EncodedObservation) -> dict[str, Any]:
    return {
        "snapshot": _snapshot_to_dict(observation.snapshot),
        "legal_action_ids": list(observation.legal_action_ids),
        "features": dict(observation.features),
        "metadata": dict(observation.metadata),
    }


def _snapshot_to_dict(snapshot: GameStateSnapshot) -> dict[str, Any]:
    return {
        "session_id": snapshot.session_id,
        "screen_state": snapshot.screen_state,
        "available_actions": list(snapshot.available_actions),
        "in_combat": snapshot.in_combat,
        "floor": snapshot.floor,
        "act": snapshot.act,
        "raw_state": dict(snapshot.raw_state),
    }


def _action_to_dict(action: ActionDecision | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "action_id": action.action_id,
        "arguments": dict(action.arguments),
        "metadata": dict(action.metadata),
    }


def _command_to_dict(command: ActionCommand | None) -> dict[str, Any] | None:
    if command is None:
        return None
    return {
        "session_id": command.session_id,
        "command": command.command,
        "arguments": dict(command.arguments),
    }
