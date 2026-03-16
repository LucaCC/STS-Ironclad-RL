from random import Random

import pytest

from sts_ironclad_rl import Action, EncounterConfig
from sts_ironclad_rl.env import CombatEnvironment
from sts_ironclad_rl.evaluation import (
    EvaluationSummary,
    evaluate_episode,
    evaluate_heuristic_policy,
    evaluate_policy,
    evaluate_random_policy,
)
from sts_ironclad_rl.evaluation.evaluator import EpisodeMetrics
from sts_ironclad_rl.evaluation.policies import HeuristicPolicy


def make_config() -> EncounterConfig:
    return EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=20,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend"),
    )


def test_evaluate_episode_is_deterministic_for_fixed_seed() -> None:
    env = CombatEnvironment(make_config())

    first = evaluate_episode(env=env, policy=HeuristicPolicy(), seed=7)
    second = evaluate_episode(env=env, policy=HeuristicPolicy(), seed=7)

    assert first == second


def test_random_policy_summary_is_reproducible_for_fixed_seeds() -> None:
    seeds = [1, 2, 3, 4]

    first = evaluate_random_policy(make_config(), seeds=seeds)
    second = evaluate_random_policy(make_config(), seeds=seeds)

    assert first == second


def test_heuristic_policy_summary_matches_expected_metrics() -> None:
    summary = evaluate_heuristic_policy(make_config(), seeds=[1, 2, 3])

    assert summary == EvaluationSummary(
        policy_name="heuristic",
        seeds=(1, 2, 3),
        episodes=(
            EpisodeMetrics(
                seed=1,
                won=True,
                total_reward=24.0,
                combat_length=7,
                damage_taken=1,
                remaining_hp=79,
            ),
            EpisodeMetrics(
                seed=2,
                won=True,
                total_reward=24.0,
                combat_length=7,
                damage_taken=2,
                remaining_hp=78,
            ),
            EpisodeMetrics(
                seed=3,
                won=True,
                total_reward=24.0,
                combat_length=7,
                damage_taken=1,
                remaining_hp=79,
            ),
        ),
        win_rate=1.0,
        mean_episode_reward=24.0,
        mean_combat_length=7.0,
        mean_damage_taken=4 / 3,
        mean_remaining_hp=236 / 3,
    )


def test_evaluate_policy_accepts_callable_policies() -> None:
    def end_turn_then_attack_policy(
        state: object,
        action_mask: dict[Action, bool],
        rng: Random,
    ) -> Action:
        del state, rng
        if action_mask[Action.END_TURN] and action_mask[Action.ATTACK] is False:
            return Action.END_TURN
        if action_mask[Action.ATTACK]:
            return Action.ATTACK
        return Action.END_TURN

    summary = evaluate_policy(make_config(), policy=end_turn_then_attack_policy, seeds=[2, 3])

    assert summary.policy_name == "callable"
    assert summary.seeds == (2, 3)
    assert len(summary.episodes) == 2


def test_evaluate_policy_rejects_empty_seed_list() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        evaluate_policy(make_config(), policy=HeuristicPolicy(), seeds=[])
