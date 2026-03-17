"""Training loops and rollout utilities for the Slay the Spire RL stack."""

from .rollout import EpisodeMetrics, EvaluationSummary, evaluate_policy, rollout_episode
from .trainer import (
    TrainingConfig,
    TrainingRunResult,
    default_encounter_config,
    run_baseline_trainer,
)

__all__ = [
    "EpisodeMetrics",
    "EvaluationSummary",
    "TrainingConfig",
    "TrainingRunResult",
    "default_encounter_config",
    "evaluate_policy",
    "rollout_episode",
    "run_baseline_trainer",
]
