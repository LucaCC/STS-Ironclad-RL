"""Benchmark configs, reporting, and artifact helpers for live-policy comparisons."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

from ..live import EvaluationSummary, summary_to_dict
from .artifacts import slugify


@dataclass(frozen=True)
class BenchmarkPolicySpec:
    """One policy entry in a live benchmark config."""

    policy_name: str
    policy_ref: str
    seed: int | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        policy_name = self.policy_name.strip()
        if not policy_name:
            raise ValueError("policy_name must not be empty")
        policy_ref = self.policy_ref.strip()
        if not policy_ref:
            raise ValueError("policy_ref must not be empty")
        if self.notes is not None and not isinstance(self.notes, str):
            raise ValueError("notes must be a string when provided")
        object.__setattr__(self, "policy_name", policy_name)
        object.__setattr__(self, "policy_ref", policy_ref)
        object.__setattr__(self, "notes", self.notes.strip() if self.notes is not None else None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": self.notes,
            "policy_name": self.policy_name,
            "policy_ref": self.policy_ref,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            policy_name=_read_required_str(payload, "policy_name"),
            policy_ref=_read_required_str(payload, "policy_ref"),
            seed=_read_optional_int(payload, "seed"),
            notes=_read_optional_str(payload, "notes"),
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    """Explicit config for a live benchmark batch over one or more policies."""

    experiment_name: str
    episode_count: int
    evaluation_case_name: str = "benchmark"
    max_steps: int = 200
    policies: tuple[BenchmarkPolicySpec, ...] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        experiment_name = self.experiment_name.strip()
        if not experiment_name:
            raise ValueError("experiment_name must not be empty")
        case_name = self.evaluation_case_name.strip()
        if not case_name:
            raise ValueError("evaluation_case_name must not be empty")
        if self.episode_count <= 0:
            raise ValueError("episode_count must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not self.policies:
            raise ValueError("policies must not be empty")

        policy_names = tuple(policy.policy_name for policy in self.policies)
        if len(set(policy_names)) != len(policy_names):
            raise ValueError("policy_name values must be unique")

        normalized_metadata = dict(self.metadata)
        try:
            json.dumps(normalized_metadata, sort_keys=True)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serializable") from exc

        object.__setattr__(self, "experiment_name", experiment_name)
        object.__setattr__(self, "evaluation_case_name", case_name)
        object.__setattr__(self, "notes", self.notes.strip() if self.notes is not None else None)
        object.__setattr__(self, "metadata", normalized_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "evaluation_case_name": self.evaluation_case_name,
            "experiment_name": self.experiment_name,
            "max_steps": self.max_steps,
            "metadata": dict(self.metadata),
            "notes": self.notes,
            "policies": [policy.to_dict() for policy in self.policies],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        policies_raw = payload.get("policies")
        if not isinstance(policies_raw, Sequence) or isinstance(policies_raw, str):
            raise ValueError("policies must be a list of policy configs")
        return cls(
            experiment_name=_read_required_str(payload, "experiment_name"),
            episode_count=_read_required_int(payload, "episode_count"),
            evaluation_case_name=str(payload.get("evaluation_case_name", "benchmark")),
            max_steps=_read_required_int(payload, "max_steps"),
            policies=tuple(_read_policy_specs(policies_raw)),
            notes=_read_optional_str(payload, "notes"),
            metadata=_read_metadata(payload.get("metadata", {})),
        )


def load_benchmark_spec(config_path: Path) -> BenchmarkSpec:
    """Load one benchmark config from a JSON file."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark config must be a JSON object")
    return BenchmarkSpec.from_dict(payload)


@dataclass(frozen=True)
class BenchmarkComparisonRow:
    """One comparable view over an evaluation summary."""

    policy_name: str
    episode_count: int
    terminal_rate: float
    interruption_rate: float
    win_rate: float
    loss_rate: float
    victory_count: int
    defeat_count: int
    mean_steps: float
    mean_total_reward: float | None
    mean_final_score: float | None
    mean_final_floor: float | None
    invalid_action_count: int | None = None
    mask_fallback_count: int | None = None
    epsilon: float | None = None
    optimization_steps: int | None = None
    outcome_counts: Mapping[str, int] = field(default_factory=dict)
    failure_counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "defeat_count": self.defeat_count,
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "failure_counts": dict(self.failure_counts),
            "interruption_rate": self.interruption_rate,
            "invalid_action_count": self.invalid_action_count,
            "loss_rate": self.loss_rate,
            "mask_fallback_count": self.mask_fallback_count,
            "mean_final_floor": self.mean_final_floor,
            "mean_final_score": self.mean_final_score,
            "mean_steps": self.mean_steps,
            "mean_total_reward": self.mean_total_reward,
            "optimization_steps": self.optimization_steps,
            "outcome_counts": dict(self.outcome_counts),
            "policy_name": self.policy_name,
            "terminal_rate": self.terminal_rate,
            "victory_count": self.victory_count,
            "win_rate": self.win_rate,
        }


@dataclass(frozen=True)
class BenchmarkComparisonReport:
    """Side-by-side benchmark report over multiple policy summaries."""

    experiment_name: str
    case_name: str
    rows: tuple[BenchmarkComparisonRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "experiment_name": self.experiment_name,
            "rows": [row.to_dict() for row in self.rows],
        }


def build_comparison_report(
    *,
    experiment_name: str,
    case_name: str,
    summaries: Sequence[EvaluationSummary],
    trainer_metrics_by_policy: Mapping[str, Mapping[str, Any]] | None = None,
) -> BenchmarkComparisonReport:
    """Build a compact comparison report from per-policy evaluation summaries."""

    trainer_metrics = dict(trainer_metrics_by_policy or {})
    rows = tuple(
        _build_comparison_row(
            summary=summary,
            trainer_metrics=trainer_metrics.get(summary.policy_name, {}),
        )
        for summary in summaries
    )
    return BenchmarkComparisonReport(
        experiment_name=experiment_name,
        case_name=case_name,
        rows=rows,
    )


def format_comparison_report(report: BenchmarkComparisonReport) -> str:
    """Render a human-readable benchmark comparison."""

    lines = [
        f"benchmark={report.experiment_name} case={report.case_name} policies={len(report.rows)}"
    ]
    for row in report.rows:
        metric_parts = [
            f"policy={row.policy_name}",
            f"win_rate={row.win_rate:.2f}",
            f"loss_rate={row.loss_rate:.2f}",
            f"completion_rate={row.terminal_rate:.2f}",
            f"interruption_rate={row.interruption_rate:.2f}",
            f"mean_steps={row.mean_steps:.2f}",
        ]
        if row.mean_total_reward is not None:
            metric_parts.append(f"mean_total_reward={row.mean_total_reward:.2f}")
        if row.mean_final_score is not None:
            metric_parts.append(f"mean_final_score={row.mean_final_score:.2f}")
        if row.mean_final_floor is not None:
            metric_parts.append(f"mean_final_floor={row.mean_final_floor:.2f}")
        if row.invalid_action_count is not None:
            metric_parts.append(f"invalid_actions={row.invalid_action_count}")
        if row.mask_fallback_count is not None:
            metric_parts.append(f"mask_fallbacks={row.mask_fallback_count}")
        if row.epsilon is not None:
            metric_parts.append(f"epsilon={row.epsilon:.3f}")
        if row.optimization_steps is not None:
            metric_parts.append(f"updates={row.optimization_steps}")
        lines.append(" ".join(metric_parts))
        lines.append(
            "  outcomes="
            + _format_counts(row.outcome_counts)
            + " failures="
            + _format_counts(row.failure_counts)
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class BenchmarkRunLayout:
    """Filesystem layout for one benchmark run."""

    run_dir: Path
    config_path: Path
    summaries_path: Path
    comparison_path: Path
    comparison_text_path: Path


@dataclass
class BenchmarkArtifactStore:
    """Write benchmark configs, summaries, and comparison reports."""

    root_dir: Path

    def create_run_layout(self, *, spec: BenchmarkSpec, started_at: datetime) -> BenchmarkRunLayout:
        run_id = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_dir = self.root_dir / slugify(spec.experiment_name)
        run_dir = _create_run_dir(experiment_dir=experiment_dir, run_id=run_id)
        return BenchmarkRunLayout(
            run_dir=run_dir,
            config_path=run_dir / "config.json",
            summaries_path=run_dir / "summaries.json",
            comparison_path=run_dir / "comparison.json",
            comparison_text_path=run_dir / "comparison.txt",
        )

    def write_run_artifacts(
        self,
        *,
        layout: BenchmarkRunLayout,
        spec: BenchmarkSpec,
        summaries: Sequence[EvaluationSummary],
        report: BenchmarkComparisonReport,
    ) -> None:
        _write_json(layout.config_path, spec.to_dict())
        _write_json(
            layout.summaries_path,
            {"summaries": [summary_to_dict(summary) for summary in summaries]},
        )
        _write_json(layout.comparison_path, report.to_dict())
        layout.comparison_text_path.write_text(
            format_comparison_report(report) + "\n",
            encoding="utf-8",
        )


def _build_comparison_row(
    *,
    summary: EvaluationSummary,
    trainer_metrics: Mapping[str, Any],
) -> BenchmarkComparisonRow:
    episode_count = summary.episode_count
    victory_count = int(summary.outcome_counts.get("victory", 0))
    defeat_count = int(summary.outcome_counts.get("defeat", 0))
    recent_metrics = trainer_metrics.get("recent_metrics", {})
    state = trainer_metrics.get("state", {})
    return BenchmarkComparisonRow(
        policy_name=summary.policy_name,
        episode_count=episode_count,
        terminal_rate=_safe_rate(summary.terminal_episode_count, episode_count),
        interruption_rate=_safe_rate(summary.interruption_count, episode_count),
        win_rate=_safe_rate(victory_count, episode_count),
        loss_rate=_safe_rate(defeat_count, episode_count),
        victory_count=victory_count,
        defeat_count=defeat_count,
        mean_steps=summary.mean_steps,
        mean_total_reward=summary.mean_total_reward,
        mean_final_score=summary.mean_final_score,
        mean_final_floor=summary.mean_final_floor,
        invalid_action_count=_read_optional_metric(
            summary.metadata.get("invalid_action_count"),
            recent_metrics.get("invalid_action_count"),
        ),
        mask_fallback_count=_read_optional_metric(
            summary.metadata.get("mask_fallback_count"),
            recent_metrics.get("mask_fallback_count"),
        ),
        epsilon=_read_optional_float(recent_metrics.get("epsilon")),
        optimization_steps=_read_optional_metric(state.get("optimization_steps")),
        outcome_counts=dict(summary.outcome_counts),
        failure_counts=dict(summary.failure_counts),
    )


def _safe_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _read_policy_specs(payloads: Sequence[Any]) -> list[BenchmarkPolicySpec]:
    specs: list[BenchmarkPolicySpec] = []
    for item in payloads:
        if not isinstance(item, Mapping):
            raise ValueError("each policy config must be a JSON object")
        specs.append(BenchmarkPolicySpec.from_dict(item))
    return specs


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


def _read_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON object")
    return dict(value)


def _read_optional_metric(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _read_optional_float(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _format_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "none"
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _create_run_dir(*, experiment_dir: Path, run_id: str) -> Path:
    candidate = experiment_dir / run_id
    suffix = 1
    while candidate.exists():
        candidate = experiment_dir / f"{run_id}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "BenchmarkArtifactStore",
    "BenchmarkComparisonReport",
    "BenchmarkComparisonRow",
    "BenchmarkPolicySpec",
    "BenchmarkRunLayout",
    "BenchmarkSpec",
    "build_comparison_report",
    "format_comparison_report",
    "load_benchmark_spec",
]
