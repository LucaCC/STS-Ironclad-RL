"""Minimal experiment runner built on top of the live rollout path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ..live import Policy, RolloutResult, RolloutRunner, summarize_rollouts
from .artifacts import ExperimentArtifactStore, ExperimentRunLayout, RunMetadata, make_run_metadata
from .specs import ExperimentSpec


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PolicyProvider(Protocol):
    """Build a policy instance for a given experiment spec."""

    def build_policy(self, spec: ExperimentSpec) -> Policy:
        """Return a policy configured for the requested experiment."""


@dataclass(frozen=True)
class ExperimentRunResult:
    """In-memory result of one collection run."""

    layout: ExperimentRunLayout
    metadata: RunMetadata
    episodes: tuple[RolloutResult, ...]

    @property
    def output_dir(self) -> Path:
        """Return the run artifact directory."""

        return self.layout.run_dir


@dataclass
class ExperimentRunner:
    """Collect rollout episodes and write replay-backed experiment artifacts."""

    rollout_runner: RolloutRunner
    artifact_store: ExperimentArtifactStore
    clock: Callable[[], datetime] = _utc_now

    def run(
        self,
        *,
        spec: ExperimentSpec,
        policy: Policy | None = None,
        policy_provider: PolicyProvider | None = None,
    ) -> ExperimentRunResult:
        """Execute a collection run through the shared rollout interface."""

        selected_policy = _resolve_policy(spec=spec, policy=policy, policy_provider=policy_provider)
        started_at = self.clock()
        layout = self.artifact_store.create_run_layout(spec=spec, started_at=started_at)
        evaluation_case = spec.to_evaluation_case()
        episodes = tuple(
            self.rollout_runner.run_episode(policy=selected_policy, evaluation_case=evaluation_case)
            for _ in range(spec.episode_count)
        )
        completed_at = self.clock()
        metadata = make_run_metadata(
            layout=layout,
            spec=spec,
            started_at=started_at,
            completed_at=completed_at,
        )
        summary = summarize_rollouts(
            results=episodes,
            policy_name=selected_policy.name,
            case_name=evaluation_case.name,
        )
        self.artifact_store.write_run_artifacts(
            layout=layout,
            spec=spec,
            metadata=metadata,
            summary=summary,
            episodes=episodes,
        )
        return ExperimentRunResult(layout=layout, metadata=metadata, episodes=episodes)


def _resolve_policy(
    *,
    spec: ExperimentSpec,
    policy: Policy | None,
    policy_provider: PolicyProvider | None,
) -> Policy:
    if policy is not None and policy_provider is not None:
        raise ValueError("provide either policy or policy_provider, not both")
    if policy is None and policy_provider is None:
        raise ValueError("either policy or policy_provider is required")

    resolved_policy = policy if policy is not None else policy_provider.build_policy(spec)
    if resolved_policy.name != spec.policy_name:
        raise ValueError(
            "spec policy_name does not match the resolved policy name: "
            f"{spec.policy_name} != {resolved_policy.name}"
        )
    return resolved_policy


__all__ = ["ExperimentRunResult", "ExperimentRunner", "PolicyProvider"]
