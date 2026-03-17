"""Evaluation helpers for deterministic combat policy comparisons.

Example:
    >>> from sts_ironclad_rl.env import EncounterConfig
    >>> from sts_ironclad_rl.evaluation import evaluate_heuristic_policy
    >>> config = EncounterConfig(
    ...     player_max_hp=80,
    ...     enemy_max_hp=20,
    ...     starting_energy=3,
    ...     draw_per_turn=5,
    ...     deck=("strike", "strike", "strike", "defend", "defend"),
    ... )
    >>> summary = evaluate_heuristic_policy(config, seeds=[1, 2, 3])
    >>> print(summary.to_pretty_text())
"""

from .evaluator import (
    EpisodeMetrics,
    EvaluationSummary,
    evaluate_episode,
    evaluate_heuristic_policy,
    evaluate_policy,
    evaluate_random_policy,
)
from .policies import HeuristicPolicy, Policy, RandomLegalPolicy

__all__ = [
    "EvaluationSummary",
    "EpisodeMetrics",
    "HeuristicPolicy",
    "Policy",
    "RandomLegalPolicy",
    "evaluate_episode",
    "evaluate_heuristic_policy",
    "evaluate_policy",
    "evaluate_random_policy",
]
