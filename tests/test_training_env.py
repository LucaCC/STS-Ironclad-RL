from sts_ironclad_rl import (
    ACTION_ORDER,
    OBSERVATION_FIELDS,
    Action,
    CombatTrainingEnv,
    EncounterConfig,
)


def make_config() -> EncounterConfig:
    return EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=20,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend"),
    )


def test_reset_returns_seeded_observation_and_legal_action_info() -> None:
    env_a = CombatTrainingEnv(make_config())
    env_b = CombatTrainingEnv(make_config())

    observation_a, info_a = env_a.reset(seed=13)
    observation_b, info_b = env_b.reset(seed=13)

    assert observation_a == observation_b
    assert info_a["seed"] == 13
    assert info_a["action_mask"] == (True, True, True)
    assert info_a["legal_action_indices"] == (0, 1, 2)
    assert info_a["action_mapping"] == {0: "attack", 1: "defend", 2: "end_turn"}
    assert tuple(info_a["observation_fields"]) == OBSERVATION_FIELDS
    assert info_a["combat_state"] == info_b["combat_state"]


def test_reset_without_explicit_seed_uses_deterministic_seed_sequence() -> None:
    env_a = CombatTrainingEnv(make_config(), seed=99)
    env_b = CombatTrainingEnv(make_config(), seed=99)

    first_obs_a, first_info_a = env_a.reset()
    first_obs_b, first_info_b = env_b.reset()
    second_obs_a, second_info_a = env_a.reset()
    second_obs_b, second_info_b = env_b.reset()

    assert first_info_a["seed"] == first_info_b["seed"]
    assert second_info_a["seed"] == second_info_b["seed"]
    assert first_obs_a == first_obs_b
    assert second_obs_a == second_obs_b


def test_step_uses_policy_action_index_mapping() -> None:
    env = CombatTrainingEnv(make_config())
    observation, info = env.reset(seed=13)

    assert len(observation) > 0
    assert ACTION_ORDER[0] is Action.ATTACK
    assert info["episode_step"] == 0

    next_observation, reward, terminated, truncated, step_info = env.step(0)

    assert next_observation[3] == 2
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert step_info["selected_action_index"] == 0
    assert step_info["episode_step"] == 1


def test_rollout_terminates_on_victory() -> None:
    env = CombatTrainingEnv(make_config())
    env.reset(seed=5)

    rewards: list[float] = []
    terminated = False
    truncated = False
    info: dict[str, object] | None = None
    for action_index in (0, 0, 2, 0, 0):
        _, reward, terminated, truncated, info = env.step(action_index)
        rewards.append(reward)
        if terminated or truncated:
            break

    assert rewards == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert terminated is True
    assert truncated is False
    assert info is not None
    assert info["terminal_reason"] == "victory"
    assert info["won"] is True
    assert info["final_enemy_hp"] == 0


def test_rollout_truncates_at_max_steps() -> None:
    env = CombatTrainingEnv(make_config(), max_steps=1)
    env.reset(seed=13)

    _, reward, terminated, truncated, info = env.step(1)

    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert info["terminal_reason"] == "max_steps"
    assert info["episode_step"] == 1
