#!/usr/bin/env python3
"""Run batches of live-policy episodes through the shared live rollout path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._live_utils import (
    build_live_episode_runner,
    instantiate_transport,
    load_object,
    load_policy,
)
from sts_ironclad_rl.live import (
    EvaluationCase,
    PolicyEvaluator,
    format_evaluation_summary,
    summary_to_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        required=True,
        help="Import path to a BridgeTransport factory, for example package.module:build_transport",
    )
    parser.add_argument(
        "--policy",
        default="simple_heuristic",
        help=(
            "Built-in policy name (simple_heuristic, random_legal) or import path module:factory"
        ),
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of live episodes to run")
    parser.add_argument("--case-name", default="live_eval", help="Label used in summary output")
    parser.add_argument("--max-steps", type=int, default=200, help="Per-episode step cap")
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
        help="Optional path for a JSON summary artifact",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")

    transport_factory = load_object(args.transport)
    policy = load_policy(args.policy, seed=args.seed)
    transport = instantiate_transport(transport_factory)
    runner = build_live_episode_runner(
        transport=transport,
        host=args.host,
        port=args.port,
        max_steps=args.max_steps,
    )
    evaluator = PolicyEvaluator(runner=runner)
    result = evaluator.evaluate(
        policy=policy,
        episode_count=args.episodes,
        evaluation_case=EvaluationCase(name=args.case_name, max_steps=args.max_steps),
    )

    print(format_evaluation_summary(result.summary))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary_to_dict(result.summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
