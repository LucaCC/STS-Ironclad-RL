from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import torch

from sts_ironclad_rl.integration import GameStateSnapshot
from sts_ironclad_rl.live import EncodedObservation, EvaluationSummary
from sts_ironclad_rl.training import (
    BenchmarkArtifactStore,
    BenchmarkPolicySpec,
    BenchmarkSpec,
    MaskedDQN,
    MaskedDQNConfig,
    build_comparison_report,
    format_comparison_report,
    load_benchmark_spec,
    load_dqn_trainer_config,
    load_trained_dqn_policy,
)


def test_benchmark_spec_round_trips_and_validates(tmp_path) -> None:
    spec = BenchmarkSpec.from_dict(
        {
            "episode_count": 3,
            "evaluation_case_name": "benchmark",
            "experiment_name": "baseline policy evaluation",
            "max_steps": 150,
            "metadata": {"wave": "wave6"},
            "policies": [
                {"policy_name": "random_legal", "policy_ref": "random_legal", "seed": 7},
                {"policy_name": "simple_heuristic", "policy_ref": "simple_heuristic"},
            ],
        }
    )

    config_path = tmp_path / "benchmark.json"
    config_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    loaded = load_benchmark_spec(config_path)

    assert loaded == spec
    assert loaded.policies[0] == BenchmarkPolicySpec(
        policy_name="random_legal",
        policy_ref="random_legal",
        seed=7,
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "episode_count": 1,
                "experiment_name": "",
                "max_steps": 10,
                "policies": [{"policy_name": "random_legal", "policy_ref": "random_legal"}],
            },
            "experiment_name",
        ),
        (
            {
                "episode_count": 1,
                "experiment_name": "x",
                "max_steps": 10,
                "policies": [],
            },
            "policies",
        ),
        (
            {
                "episode_count": 1,
                "experiment_name": "x",
                "max_steps": 10,
                "policies": [
                    {"policy_name": "random_legal", "policy_ref": "random_legal"},
                    {"policy_name": "random_legal", "policy_ref": "simple_heuristic"},
                ],
            },
            "unique",
        ),
    ],
)
def test_benchmark_spec_rejects_invalid_payloads(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BenchmarkSpec.from_dict(payload)


def test_comparison_report_formats_policy_metrics_and_artifacts(tmp_path) -> None:
    report = build_comparison_report(
        experiment_name="dqn vs baselines",
        case_name="benchmark",
        summaries=[
            EvaluationSummary(
                policy_name="random_legal",
                case_name="benchmark",
                episode_count=4,
                terminal_episode_count=2,
                interruption_count=2,
                outcome_counts={"defeat": 1, "victory": 1},
                failure_counts={"bridge_disconnect": 1},
                action_counts={"end_turn": 4},
                mean_steps=12.0,
                mean_total_reward=-0.5,
                mean_final_score=20.0,
                mean_final_floor=3.0,
                metadata={},
            ),
            EvaluationSummary(
                policy_name="masked_dqn",
                case_name="benchmark",
                episode_count=4,
                terminal_episode_count=4,
                interruption_count=0,
                outcome_counts={"defeat": 1, "victory": 3},
                failure_counts={},
                action_counts={"end_turn": 2, "play_card:0": 6},
                mean_steps=9.0,
                mean_total_reward=1.5,
                mean_final_score=45.0,
                mean_final_floor=6.0,
                metadata={"invalid_action_count": 0, "mask_fallback_count": 1},
            ),
        ],
        trainer_metrics_by_policy={
            "masked_dqn": {
                "recent_metrics": {"epsilon": 0.25},
                "state": {"optimization_steps": 120},
            }
        },
    )

    assert report.rows[0].win_rate == pytest.approx(0.25)
    assert report.rows[1].optimization_steps == 120
    assert "policy=masked_dqn win_rate=0.75" in format_comparison_report(report)

    spec = BenchmarkSpec(
        experiment_name="dqn vs baselines",
        episode_count=4,
        max_steps=150,
        policies=(
            BenchmarkPolicySpec(policy_name="random_legal", policy_ref="random_legal"),
            BenchmarkPolicySpec(
                policy_name="masked_dqn",
                policy_ref="dqn_checkpoint:artifacts/training/live_dqn/checkpoints/checkpoint_final.pt",
            ),
        ),
    )
    store = BenchmarkArtifactStore(root_dir=tmp_path)
    layout = store.create_run_layout(
        spec=spec,
        started_at=datetime(2026, 3, 17, 18, 0, tzinfo=timezone.utc),
    )
    store.write_run_artifacts(
        layout=layout,
        spec=spec,
        summaries=[
            EvaluationSummary(
                policy_name="random_legal",
                case_name="benchmark",
                episode_count=1,
                terminal_episode_count=1,
                interruption_count=0,
                outcome_counts={"victory": 1},
                failure_counts={},
                action_counts={"end_turn": 1},
                mean_steps=1.0,
            )
        ],
        report=build_comparison_report(
            experiment_name="dqn vs baselines",
            case_name="benchmark",
            summaries=[
                EvaluationSummary(
                    policy_name="random_legal",
                    case_name="benchmark",
                    episode_count=1,
                    terminal_episode_count=1,
                    interruption_count=0,
                    outcome_counts={"victory": 1},
                    failure_counts={},
                    action_counts={"end_turn": 1},
                    mean_steps=1.0,
                )
            ],
        ),
    )

    assert layout.comparison_path.exists()
    assert "random_legal" in layout.comparison_text_path.read_text(encoding="utf-8")


def test_load_dqn_trainer_config_and_checkpoint_policy(tmp_path) -> None:
    config_path = tmp_path / "trainer.json"
    config_path.write_text(
        json.dumps(
            {
                "batch_size": 16,
                "evaluation_episodes": 2,
                "network": {"hidden_sizes": [8], "observation_size": 93, "action_size": 61},
                "train_episodes": 5,
            }
        ),
        encoding="utf-8",
    )
    trainer_config = load_dqn_trainer_config(config_path)
    assert trainer_config.batch_size == 16
    assert trainer_config.network.hidden_sizes == (8,)

    model = MaskedDQN(MaskedDQNConfig(hidden_sizes=(8,)))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_config": {
                "action_size": model.config.action_size,
                "hidden_sizes": list(model.config.hidden_sizes),
                "observation_size": model.config.observation_size,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    policy = load_trained_dqn_policy(checkpoint_path, seed=3)
    observation = EncodedObservation(
        snapshot=GameStateSnapshot(
            session_id="session-1",
            screen_state="COMBAT",
            available_actions=("end_turn",),
            in_combat=True,
            floor=1,
            act=1,
            raw_state={},
        ),
        legal_action_ids=("end_turn",),
        features={},
        metadata={"structured_observation": {"in_combat": True, "combat": {}}},
    )

    assert policy.name == "masked_dqn"
    assert policy.select_action(observation).action_id == "end_turn"
