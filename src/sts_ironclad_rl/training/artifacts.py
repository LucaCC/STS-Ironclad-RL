"""Artifact layout and serialization for live experiment runs."""

from __future__ import annotations

import json
import platform
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..live import EvaluationSummary, RolloutResult, replay_entry_to_dict, summary_to_dict
from .specs import ExperimentSpec

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ExperimentRunLayout:
    """Filesystem layout for one experiment run."""

    root_dir: Path
    experiment_dir: Path
    run_dir: Path
    config_path: Path
    metadata_path: Path
    summary_path: Path
    episodes_path: Path
    trajectory_path: Path


@dataclass(frozen=True)
class DQNTrainerRunLayout:
    """Filesystem layout for one masked-DQN training run."""

    root_dir: Path
    config_path: Path
    metrics_path: Path
    evaluations_path: Path
    summary_path: Path
    checkpoints_dir: Path

    @property
    def final_checkpoint_path(self) -> Path:
        """Return the canonical final checkpoint location."""

        return self.checkpoints_dir / "checkpoint_final.pt"


@dataclass(frozen=True)
class RunMetadata:
    """Captured metadata for one experiment execution."""

    run_id: str
    experiment_name: str
    policy_name: str
    episode_count: int
    evaluation_case_name: str
    spec_fingerprint: str
    started_at: str
    completed_at: str
    duration_seconds: float
    package_version: str = __version__
    python_version: str = platform.python_version()
    platform: str = platform.platform()
    seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable view of the metadata."""

        return asdict(self)


@dataclass
class ExperimentArtifactStore:
    """Create and populate deterministic run output directories."""

    root_dir: Path

    def create_run_layout(
        self,
        *,
        spec: ExperimentSpec,
        started_at: datetime,
    ) -> ExperimentRunLayout:
        """Return the standard layout for a new run and create its directories."""

        run_id = build_run_id(spec=spec, started_at=started_at)
        experiment_dir = self.root_dir / slugify(spec.experiment_name)
        run_dir = _create_run_dir(experiment_dir=experiment_dir, run_id=run_id)
        return ExperimentRunLayout(
            root_dir=self.root_dir,
            experiment_dir=experiment_dir,
            run_dir=run_dir,
            config_path=run_dir / "config.json",
            metadata_path=run_dir / "metadata.json",
            summary_path=run_dir / "summary.json",
            episodes_path=run_dir / "episodes.jsonl",
            trajectory_path=run_dir / "trajectory.jsonl",
        )

    def write_run_artifacts(
        self,
        *,
        layout: ExperimentRunLayout,
        spec: ExperimentSpec,
        metadata: RunMetadata,
        summary: EvaluationSummary,
        episodes: tuple[RolloutResult, ...],
    ) -> None:
        """Write the canonical config, metadata, summary, and dataset files."""

        _write_json(layout.config_path, spec.to_dict())
        _write_json(layout.metadata_path, metadata.to_dict())
        _write_json(layout.summary_path, summary_to_dict(summary))
        _write_jsonl(
            layout.episodes_path,
            (_episode_payload(index=index, result=result) for index, result in enumerate(episodes)),
        )
        _write_jsonl(
            layout.trajectory_path,
            (
                _trajectory_payload(episode_index=episode_index, result=result)
                for episode_index, result in enumerate(episodes)
            ),
        )


def build_run_id(*, spec: ExperimentSpec, started_at: datetime) -> str:
    """Build a stable run directory name from time and spec content."""

    timestamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{spec.fingerprint()[:8]}"


def make_run_metadata(
    *,
    layout: ExperimentRunLayout,
    spec: ExperimentSpec,
    started_at: datetime,
    completed_at: datetime,
) -> RunMetadata:
    """Construct reproducibility metadata for one finished run."""

    run_id = layout.run_dir.name
    duration_seconds = max(0.0, (completed_at - started_at).total_seconds())
    return RunMetadata(
        run_id=run_id,
        experiment_name=spec.experiment_name,
        policy_name=spec.policy_name,
        episode_count=spec.episode_count,
        evaluation_case_name=spec.evaluation_case_name,
        spec_fingerprint=spec.fingerprint(),
        started_at=_isoformat(started_at),
        completed_at=_isoformat(completed_at),
        duration_seconds=duration_seconds,
        seed=spec.seed,
    )


def create_dqn_trainer_run_layout(root_dir: Path) -> DQNTrainerRunLayout:
    """Create the canonical masked-DQN training artifact layout."""

    root_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = root_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return DQNTrainerRunLayout(
        root_dir=root_dir,
        config_path=root_dir / "config.json",
        metrics_path=root_dir / "metrics.jsonl",
        evaluations_path=root_dir / "evaluations.jsonl",
        summary_path=root_dir / "summary.json",
        checkpoints_dir=checkpoints_dir,
    )


def resolve_dqn_training_summary_path(checkpoint_path: str | Path) -> Path:
    """Return the trainer summary path for a canonical checkpoint location."""

    checkpoint = Path(checkpoint_path)
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("checkpoint_path must live under a checkpoints/ directory")
    return checkpoint.parent.parent / "summary.json"


def slugify(value: str) -> str:
    """Normalize a human name for deterministic filesystem use."""

    normalized = _SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    return normalized or "experiment"


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _episode_payload(*, index: int, result: RolloutResult) -> dict[str, object]:
    payload = {
        "episode_index": index,
        "failure": asdict(result.failure) if result.failure is not None else None,
        "metadata": dict(result.metadata),
        "outcome": result.outcome,
        "session_id": result.session_id,
        "step_count": result.step_count,
        "terminal": result.terminal,
        "total_reward": result.total_reward,
    }
    return payload


def _trajectory_payload(*, episode_index: int, result: RolloutResult) -> list[dict[str, object]]:
    return [
        {
            "episode_index": episode_index,
            "entry": replay_entry_to_dict(entry),
            "session_id": result.session_id,
        }
        for entry in result.entries
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payloads) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            if isinstance(payload, list):
                for item in payload:
                    handle.write(json.dumps(item, sort_keys=True))
                    handle.write("\n")
                continue
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def _create_run_dir(*, experiment_dir: Path, run_id: str) -> Path:
    candidate = experiment_dir / run_id
    suffix = 1
    while candidate.exists():
        candidate = experiment_dir / f"{run_id}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


__all__ = [
    "DQNTrainerRunLayout",
    "ExperimentArtifactStore",
    "ExperimentRunLayout",
    "RunMetadata",
    "build_run_id",
    "create_dqn_trainer_run_layout",
    "make_run_metadata",
    "resolve_dqn_training_summary_path",
    "slugify",
]
