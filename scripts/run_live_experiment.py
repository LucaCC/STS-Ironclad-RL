#!/usr/bin/env python3
"""Run a replay-backed live collection experiment from a JSON spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts_ironclad_rl.live import (
    build_live_episode_runner,
    instantiate_transport,
    load_live_policy,
    load_object,
)
from sts_ironclad_rl.training import ExperimentArtifactStore, ExperimentRunner, load_experiment_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        required=True,
        help="Import path to a BridgeTransport factory, for example package.module:build_transport",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment JSON config, for example configs/experiments/random_legal_collection.json",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Root directory for experiment outputs",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help=(
            "Optional built-in policy name "
            "(simple_heuristic, random_legal), dqn_checkpoint:/path/to/checkpoint.pt, "
            "or import path module:factory"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used when loading checkpoint-backed DQN policies",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bridge host forwarded to BridgeConfig")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bridge port forwarded to BridgeConfig",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the built-in random policy",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for a JSON copy of the aggregate run summary",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    spec = load_experiment_spec(args.config)
    transport_factory = load_object(args.transport)
    transport = instantiate_transport(transport_factory)
    runner = build_live_episode_runner(
        transport=transport,
        host=args.host,
        port=args.port,
        max_steps=spec.max_steps or 200,
    )
    policy_name = args.policy or spec.policy_name
    policy = load_live_policy(
        policy_name,
        seed=args.seed,
        device=args.device,
        policy_name=spec.policy_name,
    )
    experiment_runner = ExperimentRunner(
        rollout_runner=runner,
        artifact_store=ExperimentArtifactStore(root_dir=args.artifacts_dir),
    )
    result = experiment_runner.run(spec=spec, policy=policy)

    summary_payload = json.loads(
        result.output_dir.joinpath("summary.json").read_text(encoding="utf-8")
    )
    print(f"run_dir={result.output_dir}")
    print(json.dumps(summary_payload, indent=2, sort_keys=True))

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
