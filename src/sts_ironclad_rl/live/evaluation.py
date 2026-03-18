"""Evaluation helpers for live-game policy batches."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .contracts import EvaluationCase, EvaluationSummary, Policy, RolloutResult, RolloutRunner


@dataclass(frozen=True)
class BatchEvaluationResult:
    """Policy evaluation batch built on top of rollout results."""

    summary: EvaluationSummary
    episodes: tuple[RolloutResult, ...]


@dataclass
class PolicyEvaluator:
    """Run repeated episodes through a rollout runner and summarize them."""

    runner: RolloutRunner

    def evaluate(
        self,
        *,
        policy: Policy,
        episode_count: int,
        evaluation_case: EvaluationCase | None = None,
    ) -> BatchEvaluationResult:
        results = tuple(
            self.runner.run_episode(policy=policy, evaluation_case=evaluation_case)
            for _ in range(episode_count)
        )
        summary = summarize_rollouts(
            results=results,
            policy_name=policy.name,
            case_name=evaluation_case.name if evaluation_case is not None else "default",
        )
        return BatchEvaluationResult(summary=summary, episodes=results)


def summarize_rollouts(
    *,
    results: tuple[RolloutResult, ...],
    policy_name: str,
    case_name: str,
) -> EvaluationSummary:
    outcome_counts = Counter(_outcome_label(result) for result in results)
    failure_counts = Counter(
        result.failure.kind for result in results if result.failure is not None
    )
    action_counts = Counter(
        entry.action.action_id
        for result in results
        for entry in result.entries
        if entry.action is not None
    )
    terminal_episode_count = sum(1 for result in results if result.terminal)
    interruption_count = sum(1 for result in results if not result.terminal)
    total_rewards = [result.total_reward for result in results if result.total_reward is not None]
    final_scores = [
        float(score)
        for result in results
        if (score := result.metadata.get("final_score")) is not None
        and isinstance(score, int | float)
    ]
    final_floors = [
        float(floor)
        for result in results
        if (floor := result.metadata.get("final_floor")) is not None and isinstance(floor, int)
    ]
    mean_steps = fmean([float(result.step_count) for result in results]) if results else 0.0

    return EvaluationSummary(
        policy_name=policy_name,
        case_name=case_name,
        episode_count=len(results),
        terminal_episode_count=terminal_episode_count,
        interruption_count=interruption_count,
        outcome_counts=dict(sorted(outcome_counts.items())),
        failure_counts=dict(sorted(failure_counts.items())),
        action_counts=dict(sorted(action_counts.items())),
        mean_steps=mean_steps,
        mean_total_reward=fmean(total_rewards) if total_rewards else None,
        mean_final_score=fmean(final_scores) if final_scores else None,
        mean_final_floor=fmean(final_floors) if final_floors else None,
        metadata={"terminal_rate": _safe_rate(terminal_episode_count, len(results))},
    )


def summary_to_dict(summary: EvaluationSummary) -> dict[str, Any]:
    """Return a JSON-serializable evaluation summary payload."""

    return {
        "policy_name": summary.policy_name,
        "case_name": summary.case_name,
        "episode_count": summary.episode_count,
        "terminal_episode_count": summary.terminal_episode_count,
        "interruption_count": summary.interruption_count,
        "outcome_counts": dict(summary.outcome_counts),
        "failure_counts": dict(summary.failure_counts),
        "action_counts": dict(summary.action_counts),
        "mean_steps": summary.mean_steps,
        "mean_total_reward": summary.mean_total_reward,
        "mean_final_score": summary.mean_final_score,
        "mean_final_floor": summary.mean_final_floor,
        "metadata": dict(summary.metadata),
    }


def format_evaluation_summary(summary: EvaluationSummary) -> str:
    """Render a compact text summary for CLI output."""

    lines = [
        f"policy={summary.policy_name} case={summary.case_name} episodes={summary.episode_count}",
        (
            "terminal="
            f"{summary.terminal_episode_count}/{summary.episode_count} "
            f"interrupted={summary.interruption_count} "
            f"mean_steps={summary.mean_steps:.2f}"
        ),
        (
            "rates="
            f"completion:{_safe_rate(summary.terminal_episode_count, summary.episode_count):.2f} "
            f"interruption:{_safe_rate(summary.interruption_count, summary.episode_count):.2f} "
            "victory:"
            f"{_safe_rate(summary.outcome_counts.get('victory', 0), summary.episode_count):.2f} "
            "defeat:"
            f"{_safe_rate(summary.outcome_counts.get('defeat', 0), summary.episode_count):.2f}"
        ),
        f"outcomes={_format_counts(summary.outcome_counts)}",
        f"actions={_format_counts(summary.action_counts)}",
        f"failures={_format_counts(summary.failure_counts)}",
    ]

    proxy_parts: list[str] = []
    if summary.mean_total_reward is not None:
        proxy_parts.append(f"mean_total_reward={summary.mean_total_reward:.2f}")
    if summary.mean_final_score is not None:
        proxy_parts.append(f"mean_final_score={summary.mean_final_score:.2f}")
    if summary.mean_final_floor is not None:
        proxy_parts.append(f"mean_final_floor={summary.mean_final_floor:.2f}")
    if proxy_parts:
        lines.append("proxies=" + " ".join(proxy_parts))

    policy_parts: list[str] = []
    if isinstance(summary.metadata.get("invalid_action_count"), int):
        policy_parts.append(f"invalid_action_count={summary.metadata['invalid_action_count']}")
    if isinstance(summary.metadata.get("mask_fallback_count"), int):
        policy_parts.append(f"mask_fallback_count={summary.metadata['mask_fallback_count']}")
    if policy_parts:
        lines.append("policy_metrics=" + " ".join(policy_parts))

    return "\n".join(lines)


def summary_to_json(summary: EvaluationSummary) -> str:
    """Return the summary as stable pretty-printed JSON."""

    return json.dumps(summary_to_dict(summary), indent=2, sort_keys=True)


def _safe_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _outcome_label(result: RolloutResult) -> str:
    if result.failure is not None:
        return f"failure:{result.failure.kind}"
    return result.outcome or ("terminal" if result.terminal else "interrupted")


def _format_counts(counts: dict[str, int] | Any) -> str:
    normalized = dict(counts)
    if not normalized:
        return "none"
    return ",".join(f"{key}:{normalized[key]}" for key in sorted(normalized))
