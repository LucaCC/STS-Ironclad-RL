#!/usr/bin/env python3
"""Run batches of live-policy episodes through the bridge rollout runner."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from sts_ironclad_rl.integration import BridgeConfig, BridgeTransport, LiveGameBridge
from sts_ironclad_rl.live import (
    BridgeObservationEncoder,
    CommunicationModActionContract,
    EvaluationCase,
    LiveEpisodeRunner,
    Policy,
    PolicyEvaluator,
    RandomLegalPolicy,
    RunnerConfig,
    SimpleHeuristicPolicy,
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
        default="heuristic",
        help="Built-in policy name (heuristic, random) or import path module:factory",
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

    transport_factory = _load_object(args.transport)
    policy = _load_policy(args.policy, seed=args.seed)
    transport = _instantiate_transport(transport_factory)
    bridge = LiveGameBridge(
        transport=transport,
        config=BridgeConfig(host=args.host, port=args.port),
    )
    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=BridgeObservationEncoder(),
        action_contract=CommunicationModActionContract(),
        config=RunnerConfig(max_steps=args.max_steps),
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


def _load_policy(policy_arg: str, *, seed: int | None) -> Policy:
    if policy_arg == "heuristic":
        return SimpleHeuristicPolicy()
    if policy_arg == "random":
        return RandomLegalPolicy(seed=seed)

    factory = _load_object(policy_arg)
    policy = factory()
    if not hasattr(policy, "select_action") or not hasattr(policy, "name"):
        raise TypeError("custom policy factory must return a live Policy-compatible object")
    return policy


def _instantiate_transport(factory: Any) -> BridgeTransport:
    transport = factory()
    if not isinstance(transport, BridgeTransport):
        raise TypeError("transport factory must return a BridgeTransport instance")
    return transport


def _load_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError("import path must look like package.module:attribute")
    module_name, attribute_name = import_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


if __name__ == "__main__":
    raise SystemExit(main())
