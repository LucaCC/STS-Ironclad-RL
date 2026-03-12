"""Trajectory logging helpers for the live-game bridge."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .protocol import TrajectoryEntry


class JsonlTrajectoryLogger:
    """Append trajectory entries to a JSONL file."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    @property
    def output_path(self) -> Path:
        """Return the log destination."""
        return self._output_path

    def log(self, entry: TrajectoryEntry) -> None:
        """Append one trajectory entry as a single JSON line."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), sort_keys=True))
            handle.write("\n")
