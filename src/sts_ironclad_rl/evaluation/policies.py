"""Evaluation policy exports for the milestone 1 combat slice."""

from ..agents.baseline import BaselinePolicy as Policy
from ..agents.baseline import HeuristicPolicy, RandomPolicy

RandomLegalPolicy = RandomPolicy

__all__ = ["HeuristicPolicy", "Policy", "RandomLegalPolicy"]
