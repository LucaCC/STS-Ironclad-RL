"""Structured replay IO for live-game rollouts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .contracts import ReplayEntry

REPLAY_SCHEMA_VERSION = "sts.live_replay.v1"


def _entry_to_jsonable(entry: ReplayEntry) -> dict[str, Any]:
    payload = entry.to_dict()
    payload["schema_version"] = REPLAY_SCHEMA_VERSION
    return payload


class JsonlReplayWriter:
    """Append structured replay entries to a JSONL file."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    @property
    def output_path(self) -> Path:
        """Return the replay destination."""
        return self._output_path

    def log(self, entry: ReplayEntry) -> None:
        """Append one replay entry as a single JSON line."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_entry_to_jsonable(entry), sort_keys=True))
            handle.write("\n")

    def write_all(self, entries: Iterable[ReplayEntry]) -> None:
        """Append multiple replay entries."""
        for entry in entries:
            self.log(entry)


def iter_replay_entries(input_path: Path) -> Iterator[ReplayEntry]:
    """Yield replay entries from a JSONL file."""
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            schema_version = payload.pop("schema_version", REPLAY_SCHEMA_VERSION)
            if schema_version != REPLAY_SCHEMA_VERSION:
                msg = f"unsupported replay schema version on line {line_number}: {schema_version}"
                raise ValueError(msg)
            yield ReplayEntry.from_dict(payload)


def read_replay_entries(input_path: Path) -> tuple[ReplayEntry, ...]:
    """Read all replay entries from a JSONL file."""
    return tuple(iter_replay_entries(input_path))
