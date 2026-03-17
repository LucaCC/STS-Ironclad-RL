"""Small helpers for constructing baseline policies."""

from __future__ import annotations

from collections.abc import Mapping
from random import Random
from typing import Protocol, runtime_checkable

from ..env import Action, CombatState
from .baseline import HeuristicPolicy, RandomPolicy


@runtime_checkable
class Policy(Protocol):
    """Minimal policy contract for trainer and evaluation utilities."""

    name: str

    def select_action(
        self,
        state: CombatState,
        action_mask: Mapping[Action, bool],
        rng: Random,
    ) -> Action:
        """Choose the next environment action."""


def make_policy(name: str, *, seed: int = 0) -> Policy:
    """Construct a supported baseline policy by name."""
    normalized_name = name.strip().lower()
    if normalized_name == "random":
        return RandomPolicy(rng=Random(seed))
    if normalized_name == "heuristic":
        return HeuristicPolicy()
    msg = f"unknown policy: {name}"
    raise ValueError(msg)
