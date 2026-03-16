from io import StringIO

from sts_ironclad_rl.agents import make_policy
from sts_ironclad_rl.training import TrainingConfig, evaluate_policy, run_baseline_trainer
from sts_ironclad_rl.training.trainer import default_encounter_config


def test_policy_evaluation_is_reproducible_for_same_seed() -> None:
    config = default_encounter_config()

    summary_a = evaluate_policy(
        encounter_config=config,
        policy=make_policy("random"),
        episodes=4,
        seed=17,
    )
    summary_b = evaluate_policy(
        encounter_config=config,
        policy=make_policy("random"),
        episodes=4,
        seed=17,
    )

    assert summary_a == summary_b


def test_baseline_trainer_logs_expected_metrics() -> None:
    output = StringIO()

    result = run_baseline_trainer(
        TrainingConfig(
            train_policy="heuristic",
            eval_policy="random",
            train_episodes=3,
            eval_episodes=2,
            seed=5,
        ),
        stream=output,
    )

    assert result.training_summary.episode_count == 3
    assert result.evaluation_summary.episode_count == 2
    assert result.training_summary.average_episode_reward >= 0.0
    assert 0.0 <= result.training_summary.win_rate <= 1.0
    assert result.training_summary.average_combat_length >= 1.0
    assert result.training_summary.average_remaining_hp >= 0.0
    assert result.training_summary.average_hp_delta <= 0.0

    lines = output.getvalue().strip().splitlines()
    assert len(lines) == 4
    assert '"episode_reward"' in lines[0]
    assert '"win_rate"' in lines[-1]
