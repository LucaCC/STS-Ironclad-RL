"""Shared rollout and evaluation helpers for the milestone 1 combat slice."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..agents import BaselinePolicy
from ..env import (
    ACTION_ORDER,
    CombatState,
    CombatTrainingEnv,
    EncounterConfig,
    action_to_index,
)


@dataclass(frozen=True)
class EpisodeMetrics:
    """Per-episode metrics for a single combat rollout."""

    policy_name: str
    episode_index: int
    episode_seed: int
    total_reward: float
    won: bool
    combat_length: int
    remaining_hp: int
    hp_delta: int


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics across a set of deterministic episodes."""

    policy_name: str
    episode_count: int
    average_episode_reward: float
    win_rate: float
    average_combat_length: float
    average_remaining_hp: float
    average_hp_delta: float
    episodes: tuple[EpisodeMetrics, ...]


def rollout_episode(
    *,
    encounter_config: EncounterConfig,
    policy: BaselinePolicy,
    episode_index: int,
    episode_seed: int,
    max_actions: int = 100,
) -> EpisodeMetrics:
    """Run one deterministic combat episode for a policy."""
    env = CombatTrainingEnv(encounter_config, max_steps=max_actions, seed=episode_seed)
    _, info = env.reset(seed=episode_seed)
    state = _combat_state_from_info(info)
    starting_hp = state.player.hp
    total_reward = 0.0

    while True:
        action = policy.choose_action(state, _decode_action_mask(info["action_mask"]))
        _, reward, terminated, truncated, info = env.step(action_to_index(action))
        total_reward += reward
        state = _combat_state_from_info(info)
        if terminated or truncated:
            break

    return EpisodeMetrics(
        policy_name=type(policy).__name__,
        episode_index=episode_index,
        episode_seed=episode_seed,
        total_reward=total_reward,
        won=state.enemy.hp == 0 and state.player.hp > 0,
        combat_length=state.turn,
        remaining_hp=state.player.hp,
        hp_delta=state.player.hp - starting_hp,
    )


def evaluate_policy(
    *,
    encounter_config: EncounterConfig,
    policy: BaselinePolicy,
    episodes: int,
    seed: int,
    max_actions_per_episode: int = 100,
) -> EvaluationSummary:
    """Evaluate a policy across a deterministic sequence of episodes."""
    if episodes <= 0:
        msg = "episodes must be positive"
        raise ValueError(msg)

    episode_metrics = tuple(
        rollout_episode(
            encounter_config=encounter_config,
            policy=policy,
            episode_index=episode_index,
            episode_seed=seed + episode_index,
            max_actions=max_actions_per_episode,
        )
        for episode_index in range(episodes)
    )

    return EvaluationSummary(
        policy_name=type(policy).__name__,
        episode_count=episodes,
        average_episode_reward=_mean(metric.total_reward for metric in episode_metrics),
        win_rate=_mean(1.0 if metric.won else 0.0 for metric in episode_metrics),
        average_combat_length=_mean(metric.combat_length for metric in episode_metrics),
        average_remaining_hp=_mean(metric.remaining_hp for metric in episode_metrics),
        average_hp_delta=_mean(metric.hp_delta for metric in episode_metrics),
        episodes=episode_metrics,
    )


def _mean(values: Iterable[float]) -> float:
    series = tuple(float(value) for value in values)
    return sum(series) / len(series)


def _decode_action_mask(encoded_mask: object) -> dict:
    mask = tuple(encoded_mask)
    return {action: mask[index] for index, action in enumerate(ACTION_ORDER)}


def _combat_state_from_info(info: dict[str, object]) -> CombatState:
    combat_state = info["combat_state"]
    if not isinstance(combat_state, CombatState):
        msg = "training env info did not include CombatState"
        raise TypeError(msg)
    return combat_state
