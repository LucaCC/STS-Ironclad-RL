#!/usr/bin/env python3
"""Train a minimal masked-DQN baseline through the shared live rollout path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._live_utils import build_live_episode_runner, instantiate_transport, load_object
from sts_ironclad_rl.training import DQNTrainer, DQNTrainerConfig, EpsilonSchedule, MaskedDQNConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        required=True,
        help="Import path to a BridgeTransport factory, for example package.module:build_transport",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training/live_dqn"),
        help="Directory for trainer config, metrics, summaries, and checkpoints",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Training episodes to collect")
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=3,
        help="Greedy evaluation episodes per evaluation batch",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="Per-episode step cap")
    parser.add_argument("--replay-size", type=int, default=10000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon for collection",
    )
    parser.add_argument(
        "--epsilon-final",
        type=float,
        default=0.05,
        help="Final epsilon after linear decay",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=10000,
        help="Environment steps used for linear epsilon decay",
    )
    parser.add_argument(
        "--target-update-frequency",
        type=int,
        default=100,
        help="Optimizer steps between hard target-network syncs",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Environment steps to collect before optimization starts",
    )
    parser.add_argument(
        "--evaluation-cadence",
        type=int,
        default=10,
        help="Training episodes between greedy evaluation batches",
    )
    parser.add_argument(
        "--checkpoint-cadence",
        type=int,
        default=10,
        help="Training episodes between checkpoint writes",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=(128, 128),
        help="MLP hidden layer sizes for the online and target networks",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--host", default="127.0.0.1", help="Bridge host forwarded to BridgeConfig")
    parser.add_argument(
        "--port", type=int, default=8080, help="Bridge port forwarded to BridgeConfig"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    transport_factory = load_object(args.transport)
    transport = instantiate_transport(transport_factory)
    runner = build_live_episode_runner(
        transport=transport,
        host=args.host,
        port=args.port,
        max_steps=args.max_steps,
    )
    trainer = DQNTrainer(
        rollout_runner=runner,
        config=DQNTrainerConfig(
            train_episodes=args.episodes,
            evaluation_episodes=args.evaluation_episodes,
            max_steps_per_episode=args.max_steps,
            replay_buffer_size=args.replay_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_schedule=EpsilonSchedule(
                initial=args.epsilon_start,
                final=args.epsilon_final,
                decay_steps=args.epsilon_decay_steps,
            ),
            target_update_frequency=args.target_update_frequency,
            warmup_steps=args.warmup_steps,
            evaluation_cadence=args.evaluation_cadence,
            checkpoint_cadence=args.checkpoint_cadence,
            network=MaskedDQNConfig(hidden_sizes=tuple(args.hidden_sizes)),
            seed=args.seed,
        ),
    )
    result = trainer.train(output_dir=args.output_dir)
    print(f"output_dir={result.output_dir}")
    print(json.dumps(trainer.training_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
