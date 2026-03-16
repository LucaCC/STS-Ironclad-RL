"""Deterministic evaluation helpers built on the training-facing rollout path."""

from __future__ import annotations

from collections.abc import Iterable

from ..agents import HeuristicPolicy, RandomPolicy
from ..env import EncounterConfig
from ..training.rollout import (
    EpisodeMetrics,
    EvaluationSummary,
)
from ..training.rollout import (
    evaluate_policy as evaluate_rollouts,
)
from ..training.rollout import rollout_episode as evaluate_episode


def evaluate_random_policy(
    config: EncounterConfig,
    seeds: Iterable[int],
    *,
    max_actions_per_episode: int = 100,
) -> EvaluationSummary:
    """Evaluate the built-in random legal policy over seeded episodes."""
    return evaluate_policy(
        config,
        policy=RandomPolicy(),
        seeds=seeds,
        max_actions_per_episode=max_actions_per_episode,
    )


def evaluate_heuristic_policy(
    config: EncounterConfig,
    seeds: Iterable[int],
    *,
    max_actions_per_episode: int = 100,
) -> EvaluationSummary:
    """Evaluate the built-in heuristic policy over seeded episodes."""
    return evaluate_policy(
        config,
        policy=HeuristicPolicy(),
        seeds=seeds,
        max_actions_per_episode=max_actions_per_episode,
    )


def evaluate_policy(
    config: EncounterConfig,
    policy: object,
    seeds: Iterable[int],
    *,
    max_actions_per_episode: int = 100,
) -> EvaluationSummary:
    """Evaluate a policy reproducibly over an explicit seed set."""
    return evaluate_rollouts(
        encounter_config=config,
        policy=policy,
        seeds=seeds,
        max_actions_per_episode=max_actions_per_episode,
    )


__all__ = [
    "EpisodeMetrics",
    "EvaluationSummary",
    "evaluate_episode",
    "evaluate_heuristic_policy",
    "evaluate_policy",
    "evaluate_random_policy",
]
