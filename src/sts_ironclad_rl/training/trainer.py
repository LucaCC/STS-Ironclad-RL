"""Minimal baseline-oriented trainer scaffold for milestone 1."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TextIO

from ..agents import make_policy
from ..env import EncounterConfig
from .rollout import EvaluationSummary, evaluate_policy


def default_encounter_config() -> EncounterConfig:
    """Return the first trainable deterministic combat slice."""
    return EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=20,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend"),
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the initial baseline trainer scaffold."""

    train_policy: str = "heuristic"
    eval_policy: str | None = None
    train_episodes: int = 10
    eval_episodes: int = 5
    seed: int = 0
    log_every: int = 1
    max_actions_per_episode: int = 100
    encounter_config: EncounterConfig = field(default_factory=default_encounter_config)

    def __post_init__(self) -> None:
        if self.train_episodes <= 0:
            msg = "train_episodes must be positive"
            raise ValueError(msg)
        if self.eval_episodes <= 0:
            msg = "eval_episodes must be positive"
            raise ValueError(msg)
        if self.log_every <= 0:
            msg = "log_every must be positive"
            raise ValueError(msg)
        if self.max_actions_per_episode <= 0:
            msg = "max_actions_per_episode must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrainingRunResult:
    """Outputs from a baseline trainer run."""

    config: TrainingConfig
    training_summary: EvaluationSummary
    evaluation_summary: EvaluationSummary


def run_baseline_trainer(
    config: TrainingConfig,
    stream: TextIO | None = None,
) -> TrainingRunResult:
    """Run the initial placeholder trainer over deterministic baseline episodes."""
    train_policy = make_policy(config.train_policy, seed=config.seed)
    eval_policy = make_policy(
        config.eval_policy or config.train_policy,
        seed=config.seed + config.train_episodes,
    )

    training_summary = evaluate_policy(
        encounter_config=config.encounter_config,
        policy=train_policy,
        seeds=range(config.seed, config.seed + config.train_episodes),
        max_actions_per_episode=config.max_actions_per_episode,
    )
    evaluation_summary = evaluate_policy(
        encounter_config=config.encounter_config,
        policy=eval_policy,
        seeds=range(
            config.seed + config.train_episodes,
            config.seed + config.train_episodes + config.eval_episodes,
        ),
        max_actions_per_episode=config.max_actions_per_episode,
    )

    if stream is not None:
        _emit_logs(
            stream=stream, summary=training_summary, label="train", log_every=config.log_every
        )
        _emit_summary(stream=stream, summary=evaluation_summary, label="eval")

    return TrainingRunResult(
        config=config,
        training_summary=training_summary,
        evaluation_summary=evaluation_summary,
    )


def _emit_logs(
    *,
    stream: TextIO,
    summary: EvaluationSummary,
    label: str,
    log_every: int,
) -> None:
    for episode_index, episode in enumerate(summary.episodes, start=1):
        if episode_index % log_every != 0:
            continue
        payload = {
            "phase": label,
            "policy": summary.policy_name,
            "episode": episode_index,
            "seed": episode.seed,
            "episode_reward": episode.total_reward,
            "won": episode.won,
            "combat_length": episode.combat_length,
            "remaining_hp": episode.remaining_hp,
            "hp_delta": episode.hp_delta,
        }
        stream.write(f"{json.dumps(payload, sort_keys=True)}\n")


def _emit_summary(*, stream: TextIO, summary: EvaluationSummary, label: str) -> None:
    payload = {
        "phase": label,
        "policy": summary.policy_name,
        "episodes": summary.episode_count,
        "avg_episode_reward": summary.average_episode_reward,
        "win_rate": summary.win_rate,
        "avg_combat_length": summary.average_combat_length,
        "avg_remaining_hp": summary.average_remaining_hp,
        "avg_hp_delta": summary.average_hp_delta,
    }
    stream.write(f"{json.dumps(payload, sort_keys=True)}\n")


def main(argv: list[str] | None = None) -> int:
    """Run the baseline trainer from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-policy", default="heuristic", choices=("heuristic", "random"))
    parser.add_argument("--eval-policy", default=None, choices=("heuristic", "random"))
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-actions-per-episode", type=int, default=100)
    args = parser.parse_args(argv)

    config = TrainingConfig(
        train_policy=args.train_policy,
        eval_policy=args.eval_policy,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        log_every=args.log_every,
        max_actions_per_episode=args.max_actions_per_episode,
    )
    run_baseline_trainer(config=config, stream=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
