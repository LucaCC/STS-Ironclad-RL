from sts_ironclad_rl import Action, CombatEnvironment, EncounterConfig, InvalidActionError


def make_config() -> EncounterConfig:
    return EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=20,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend"),
    )


def test_reset_is_deterministic_for_same_seed() -> None:
    env_a = CombatEnvironment(make_config())
    env_b = CombatEnvironment(make_config())

    state_a = env_a.reset(seed=13)
    state_b = env_b.reset(seed=13)

    assert state_a == state_b


def test_invalid_action_is_rejected_when_card_not_in_hand() -> None:
    env = CombatEnvironment(
        EncounterConfig(
            player_max_hp=80,
            enemy_max_hp=20,
            starting_energy=3,
            draw_per_turn=1,
            deck=("strike",),
        )
    )
    env.reset(seed=1)

    try:
        env.step(Action.DEFEND)
    except InvalidActionError as exc:
        assert "illegal action" in str(exc)
    else:
        raise AssertionError("expected illegal action to raise InvalidActionError")


def test_action_mask_reflects_energy_and_hand() -> None:
    env = CombatEnvironment(make_config())
    env.reset(seed=13)

    first_state = env.step(Action.ATTACK).state
    second_state = env.step(Action.ATTACK).state
    third_state = env.step(Action.ATTACK).state

    assert first_state.energy == 2
    assert second_state.energy == 1
    assert third_state.energy == 0
    assert env.action_mask()[Action.ATTACK] is False
    assert env.action_mask()[Action.DEFEND] is False
    assert env.action_mask()[Action.END_TURN] is True


def test_short_rollout_completes_deterministically() -> None:
    env = CombatEnvironment(make_config())
    env.reset(seed=5)

    rewards: list[float] = []
    actions = [
        Action.ATTACK,
        Action.ATTACK,
        Action.END_TURN,
        Action.ATTACK,
        Action.ATTACK,
    ]

    done = False
    for action in actions:
        result = env.step(action)
        rewards.append(result.reward)
        done = result.done
        if done:
            break

    assert rewards == [6.0, 6.0, 0.0, 6.0, 6.0]
    assert done is True
