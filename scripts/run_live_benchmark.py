#!/usr/bin/env python3
"""Run a live benchmark batch over baseline and optional DQN checkpoint policies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts._live_utils import (
    build_live_episode_runner,
    instantiate_transport,
    load_object,
    load_policy,
)
from sts_ironclad_rl.live import EvaluationCase, PolicyEvaluator
from sts_ironclad_rl.training import (
    BenchmarkArtifactStore,
    build_comparison_report,
    format_comparison_report,
    load_benchmark_spec,
)


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
        help="Benchmark JSON config, for example configs/benchmarks/baseline_eval.json",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/benchmarks"),
        help="Root directory for benchmark outputs",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bridge host forwarded to BridgeConfig")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bridge port forwarded to BridgeConfig",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used when loading checkpoint-backed DQN policies",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = load_benchmark_spec(args.config)
    transport_factory = load_object(args.transport)
    transport = instantiate_transport(transport_factory)
    runner = build_live_episode_runner(
        transport=transport,
        host=args.host,
        port=args.port,
        max_steps=spec.max_steps,
    )
    evaluator = PolicyEvaluator(runner=runner)
    summaries = []
    trainer_metrics_by_policy: dict[str, dict[str, object]] = {}

    for policy_spec in spec.policies:
        policy = load_policy(
            policy_spec.policy_ref,
            seed=policy_spec.seed,
            device=args.device,
            policy_name=policy_spec.policy_name,
        )
        evaluation = evaluator.evaluate(
            policy=policy,
            episode_count=spec.episode_count,
            evaluation_case=EvaluationCase(
                name=spec.evaluation_case_name,
                max_steps=spec.max_steps,
                metadata=dict(spec.metadata),
            ),
        )
        summary = evaluation.summary
        if hasattr(policy, "stats"):
            stats = policy.stats()
            summary = summary.__class__(
                policy_name=summary.policy_name,
                case_name=summary.case_name,
                episode_count=summary.episode_count,
                terminal_episode_count=summary.terminal_episode_count,
                interruption_count=summary.interruption_count,
                outcome_counts=summary.outcome_counts,
                failure_counts=summary.failure_counts,
                action_counts=summary.action_counts,
                mean_steps=summary.mean_steps,
                mean_total_reward=summary.mean_total_reward,
                mean_final_score=summary.mean_final_score,
                mean_final_floor=summary.mean_final_floor,
                metadata={
                    **dict(summary.metadata),
                    "invalid_action_count": stats.invalid_action_count,
                    "mask_fallback_count": stats.mask_fallback_count,
                },
            )
        summaries.append(summary)

        trainer_summary_path = _trainer_summary_path(policy_spec.policy_ref)
        if trainer_summary_path is not None and trainer_summary_path.exists():
            trainer_metrics_by_policy[policy_spec.policy_name] = json.loads(
                trainer_summary_path.read_text(encoding="utf-8")
            )

    report = build_comparison_report(
        experiment_name=spec.experiment_name,
        case_name=spec.evaluation_case_name,
        summaries=summaries,
        trainer_metrics_by_policy=trainer_metrics_by_policy,
    )
    started_at = datetime.now(timezone.utc)
    store = BenchmarkArtifactStore(root_dir=args.artifacts_dir)
    layout = store.create_run_layout(spec=spec, started_at=started_at)
    store.write_run_artifacts(layout=layout, spec=spec, summaries=summaries, report=report)

    print(f"run_dir={layout.run_dir}")
    print(format_comparison_report(report))
    return 0


def _trainer_summary_path(policy_ref: str) -> Path | None:
    if not policy_ref.startswith("dqn_checkpoint:"):
        return None
    checkpoint_path = Path(policy_ref.split(":", maxsplit=1)[1])
    return checkpoint_path.parent.parent / "summary.json"


if __name__ == "__main__":
    raise SystemExit(main())
