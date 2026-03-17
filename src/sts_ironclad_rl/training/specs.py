"""Lightweight experiment specs for live rollout collection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from ..live import EvaluationCase


@dataclass(frozen=True)
class ExperimentSpec:
    """Explicit configuration for one data-collection experiment run."""

    experiment_name: str
    policy_name: str
    episode_count: int
    evaluation_case_name: str = "collection"
    max_steps: int | None = None
    seed: int | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        experiment_name = self.experiment_name.strip()
        if not experiment_name:
            raise ValueError("experiment_name must not be empty")

        policy_name = self.policy_name.strip()
        if not policy_name:
            raise ValueError("policy_name must not be empty")

        evaluation_case_name = self.evaluation_case_name.strip()
        if not evaluation_case_name:
            raise ValueError("evaluation_case_name must not be empty")

        if self.episode_count <= 0:
            raise ValueError("episode_count must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")

        if any(not isinstance(tag, str) for tag in self.tags):
            raise ValueError("tags must contain only strings")
        normalized_tags = tuple(tag.strip() for tag in self.tags if tag.strip())

        if self.notes is not None and not isinstance(self.notes, str):
            raise ValueError("notes must be a string when provided")

        normalized_metadata = dict(self.metadata)
        try:
            json.dumps(normalized_metadata, sort_keys=True)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serializable") from exc

        object.__setattr__(self, "experiment_name", experiment_name)
        object.__setattr__(self, "policy_name", policy_name)
        object.__setattr__(self, "evaluation_case_name", evaluation_case_name)
        object.__setattr__(self, "notes", self.notes.strip() if self.notes is not None else None)
        object.__setattr__(self, "tags", normalized_tags)
        object.__setattr__(self, "metadata", normalized_metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable form of the spec."""

        return {
            "episode_count": self.episode_count,
            "evaluation_case_name": self.evaluation_case_name,
            "experiment_name": self.experiment_name,
            "max_steps": self.max_steps,
            "metadata": dict(self.metadata),
            "notes": self.notes,
            "policy_name": self.policy_name,
            "seed": self.seed,
            "tags": list(self.tags),
        }

    def fingerprint(self) -> str:
        """Return a stable content hash for the spec."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_evaluation_case(self) -> EvaluationCase:
        """Translate the spec into the shared rollout case contract."""

        return EvaluationCase(
            name=self.evaluation_case_name,
            max_steps=self.max_steps,
            metadata={"experiment_name": self.experiment_name, **dict(self.metadata)},
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Build a validated spec from a raw config mapping."""

        return cls(
            experiment_name=_read_required_str(payload, "experiment_name"),
            policy_name=_read_required_str(payload, "policy_name"),
            episode_count=_read_required_int(payload, "episode_count"),
            evaluation_case_name=str(payload.get("evaluation_case_name", "collection")),
            max_steps=_read_optional_int(payload, "max_steps"),
            seed=_read_optional_int(payload, "seed"),
            notes=_read_optional_str(payload, "notes"),
            tags=_read_tags(payload.get("tags", ())),
            metadata=_read_metadata(payload.get("metadata", {})),
        )


def load_experiment_spec(config_path: Path) -> ExperimentSpec:
    """Load one experiment spec from a JSON config file."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("experiment config must be a JSON object")
    return ExperimentSpec.from_dict(payload)


def _read_required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _read_optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    return value


def _read_required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _read_optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer when provided")
    return value


def _read_tags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("tags must be a list of strings")
    if any(not isinstance(tag, str) for tag in value):
        raise ValueError("tags must contain only strings")
    return tuple(value)


def _read_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON object")
    return dict(value)
