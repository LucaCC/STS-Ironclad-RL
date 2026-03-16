"""Deterministic evaluation utilities for the milestone 1 combat slice."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from random import Random
from typing import Any

from ..env import Action, CombatEnvironment, CombatState, EncounterConfig
from .policies import HeuristicPolicy, Policy, RandomLegalPolicy

PolicyLike = Policy | Callable[[CombatState, Mapping[Action, bool], Random], Action]


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics collected from a single seeded combat rollout."""

    seed: int
    won: bool
    total_reward: float
    combat_length: int
    damage_taken: int
    remaining_hp: int


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics over multiple deterministic episodes."""

    policy_name: str
    seeds: tuple[int, ...]
    episodes: tuple[EpisodeMetrics, ...]
    win_rate: float
    mean_episode_reward: float
    mean_combat_length: float
    mean_damage_taken: float
    mean_remaining_hp: float

    def to_pretty_text(self) -> str:
        """Render a compact human-readable summary."""
        return (
            f"policy={self.policy_name}\n"
            f"episodes={len(self.episodes)} seeds={list(self.seeds)}\n"
            f"win_rate={self.win_rate:.3f}\n"
            f"mean_episode_reward={self.mean_episode_reward:.3f}\n"
            f"mean_combat_length={self.mean_combat_length:.3f}\n"
            f"mean_damage_taken={self.mean_damage_taken:.3f}\n"
            f"mean_remaining_hp={self.mean_remaining_hp:.3f}"
        )


def evaluate_random_policy(config: EncounterConfig, seeds: Iterable[int]) -> EvaluationSummary:
    """Evaluate the built-in random legal policy over seeded episodes."""
    return evaluate_policy(config=config, policy=RandomLegalPolicy(), seeds=seeds)


def evaluate_heuristic_policy(config: EncounterConfig, seeds: Iterable[int]) -> EvaluationSummary:
    """Evaluate the built-in heuristic policy over seeded episodes."""
    return evaluate_policy(config=config, policy=HeuristicPolicy(), seeds=seeds)


def evaluate_policy(
    config: EncounterConfig,
    policy: PolicyLike,
    seeds: Iterable[int],
) -> EvaluationSummary:
    """Evaluate a policy reproducibly over the provided seeds."""
    normalized_policy = _normalize_policy(policy)
    episode_seeds = tuple(seeds)
    if not episode_seeds:
        msg = "at least one seed is required for evaluation"
        raise ValueError(msg)

    env = CombatEnvironment(config)
    episodes = tuple(
        evaluate_episode(env=env, policy=normalized_policy, seed=seed) for seed in episode_seeds
    )
    episode_count = len(episodes)
    return EvaluationSummary(
        policy_name=_policy_name(normalized_policy),
        seeds=episode_seeds,
        episodes=episodes,
        win_rate=sum(episode.won for episode in episodes) / episode_count,
        mean_episode_reward=sum(episode.total_reward for episode in episodes) / episode_count,
        mean_combat_length=sum(episode.combat_length for episode in episodes) / episode_count,
        mean_damage_taken=sum(episode.damage_taken for episode in episodes) / episode_count,
        mean_remaining_hp=sum(episode.remaining_hp for episode in episodes) / episode_count,
    )


def evaluate_episode(env: CombatEnvironment, policy: PolicyLike, seed: int) -> EpisodeMetrics:
    """Run a single deterministic rollout and return episode metrics."""
    normalized_policy = _normalize_policy(policy)
    state = env.reset(seed=seed)
    total_reward = 0.0
    combat_length = 0
    rng = Random(seed)

    while state.player.hp > 0 and state.enemy.hp > 0:
        action_mask = env.action_mask()
        action = normalized_policy.select_action(state=state, action_mask=action_mask, rng=rng)
        if not action_mask.get(action, False):
            msg = f"policy produced illegal action: {action}"
            raise ValueError(msg)
        result = env.step(action)
        total_reward += result.reward
        combat_length += 1
        state = result.state
        if result.done:
            break

    return EpisodeMetrics(
        seed=seed,
        won=state.enemy.hp == 0,
        total_reward=total_reward,
        combat_length=combat_length,
        damage_taken=state.player.max_hp - state.player.hp,
        remaining_hp=state.player.hp,
    )


def _normalize_policy(policy: PolicyLike) -> Policy:
    if hasattr(policy, "select_action"):
        return policy
    if hasattr(policy, "choose_action"):
        return _ChooseActionPolicy(policy)
    if callable(policy):
        return _CallablePolicy(policy)
    msg = "policy must implement select_action(...) or be a compatible callable"
    raise TypeError(msg)


def _policy_name(policy: Policy) -> str:
    return getattr(policy, "name", type(policy).__name__)


@dataclass(frozen=True)
class _CallablePolicy:
    """Adapter for direct callables that match the evaluator signature."""

    fn: Callable[[CombatState, Mapping[Action, bool], Random], Action]
    name: str = "callable"

    def select_action(
        self,
        state: CombatState,
        action_mask: Mapping[Action, bool],
        rng: Random,
    ) -> Action:
        return self.fn(state, action_mask, rng)


@dataclass(frozen=True)
class _ChooseActionPolicy:
    """Adapter for baseline policies that expose choose_action(...)."""

    policy: Any

    @property
    def name(self) -> str:
        return getattr(self.policy, "name", type(self.policy).__name__)

    def select_action(
        self,
        state: CombatState,
        action_mask: Mapping[Action, bool],
        rng: Random,
    ) -> Action:
        del rng
        return self.policy.choose_action(state, action_mask)
