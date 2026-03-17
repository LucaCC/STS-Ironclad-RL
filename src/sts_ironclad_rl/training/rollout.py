"""Canonical rollout and evaluation helpers for the milestone 1 combat slice."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from random import Random
from typing import Any

from ..agents import BaselinePolicy
from ..env import (
    ACTION_ORDER,
    Action,
    CombatState,
    CombatTrainingEnv,
    EncounterConfig,
    action_to_index,
)

PolicyLike = BaselinePolicy | Callable[[CombatState, Mapping[Action, bool], Random], Action] | Any


@dataclass(frozen=True)
class EpisodeMetrics:
    """Per-episode metrics for a single deterministic combat rollout."""

    seed: int
    won: bool
    total_reward: float
    combat_length: int
    damage_taken: int
    remaining_hp: int

    @property
    def episode_seed(self) -> int:
        """Compatibility alias for training log code."""
        return self.seed

    @property
    def hp_delta(self) -> int:
        """Return HP change over the episode from full health."""
        return -self.damage_taken


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics across a deterministic seeded evaluation set."""

    policy_name: str
    seeds: tuple[int, ...]
    episodes: tuple[EpisodeMetrics, ...]
    win_rate: float
    mean_episode_reward: float
    mean_combat_length: float
    mean_damage_taken: float
    mean_remaining_hp: float

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def average_episode_reward(self) -> float:
        return self.mean_episode_reward

    @property
    def average_combat_length(self) -> float:
        return self.mean_combat_length

    @property
    def average_remaining_hp(self) -> float:
        return self.mean_remaining_hp

    @property
    def average_hp_delta(self) -> float:
        return -self.mean_damage_taken

    def to_pretty_text(self) -> str:
        """Render a compact human-readable summary."""
        return (
            f"policy={self.policy_name}\n"
            f"episodes={self.episode_count} seeds={list(self.seeds)}\n"
            f"win_rate={self.win_rate:.3f}\n"
            f"mean_episode_reward={self.mean_episode_reward:.3f}\n"
            f"mean_combat_length={self.mean_combat_length:.3f}\n"
            f"mean_damage_taken={self.mean_damage_taken:.3f}\n"
            f"mean_remaining_hp={self.mean_remaining_hp:.3f}"
        )


def rollout_episode(
    *,
    encounter_config: EncounterConfig,
    policy: PolicyLike,
    seed: int,
    max_actions: int = 100,
) -> EpisodeMetrics:
    """Run one deterministic combat episode for a policy."""
    normalized_policy = _normalize_policy(policy)
    env = CombatTrainingEnv(encounter_config, max_steps=max_actions, seed=seed)
    _, info = env.reset(seed=seed)
    state = _combat_state_from_info(info)
    total_reward = 0.0
    rng = Random(seed)
    actions_taken = 0

    while True:
        action = normalized_policy.select_action(
            state=state,
            action_mask=_decode_action_mask(info["action_mask"]),
            rng=rng,
        )
        _, reward, terminated, truncated, info = env.step(action_to_index(action))
        total_reward += reward
        actions_taken += 1
        state = _combat_state_from_info(info)
        if terminated or truncated:
            break

    return EpisodeMetrics(
        seed=seed,
        won=state.enemy.hp == 0 and state.player.hp > 0,
        total_reward=total_reward,
        combat_length=actions_taken,
        damage_taken=state.player.max_hp - state.player.hp,
        remaining_hp=state.player.hp,
    )


def evaluate_policy(
    *,
    encounter_config: EncounterConfig,
    policy: PolicyLike,
    seeds: Iterable[int],
    max_actions_per_episode: int = 100,
) -> EvaluationSummary:
    """Evaluate a policy over an explicit deterministic seed set."""
    episode_seeds = tuple(seeds)
    if not episode_seeds:
        msg = "at least one seed is required for evaluation"
        raise ValueError(msg)

    normalized_policy = _normalize_policy(policy)
    episode_metrics = tuple(
        rollout_episode(
            encounter_config=encounter_config,
            policy=normalized_policy,
            seed=seed,
            max_actions=max_actions_per_episode,
        )
        for seed in episode_seeds
    )

    return EvaluationSummary(
        policy_name=_policy_name(normalized_policy),
        seeds=episode_seeds,
        episodes=episode_metrics,
        win_rate=_mean(1.0 if metric.won else 0.0 for metric in episode_metrics),
        mean_episode_reward=_mean(metric.total_reward for metric in episode_metrics),
        mean_combat_length=_mean(metric.combat_length for metric in episode_metrics),
        mean_damage_taken=_mean(metric.damage_taken for metric in episode_metrics),
        mean_remaining_hp=_mean(metric.remaining_hp for metric in episode_metrics),
    )


def _mean(values: Iterable[float]) -> float:
    series = tuple(float(value) for value in values)
    return sum(series) / len(series)


def _decode_action_mask(encoded_mask: object) -> dict[Action, bool]:
    mask = tuple(encoded_mask)
    return {action: mask[index] for index, action in enumerate(ACTION_ORDER)}


def _combat_state_from_info(info: dict[str, object]) -> CombatState:
    combat_state = info["combat_state"]
    if not isinstance(combat_state, CombatState):
        msg = "training env info did not include CombatState"
        raise TypeError(msg)
    return combat_state


def _normalize_policy(policy: PolicyLike) -> Any:
    if hasattr(policy, "select_action"):
        return policy
    if hasattr(policy, "choose_action"):
        return _ChooseActionPolicy(policy)
    if callable(policy):
        return _CallablePolicy(policy)
    msg = "policy must implement select_action(...) or be a compatible callable"
    raise TypeError(msg)


def _policy_name(policy: Any) -> str:
    return getattr(policy, "name", type(policy).__name__)


@dataclass(frozen=True)
class _CallablePolicy:
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
