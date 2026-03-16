from sts_ironclad_rl import (
    CombatEnvironment,
    EncounterConfig,
    EndTurnAction,
    InvalidActionError,
    MonsterConfig,
    MonsterIntent,
    PlayCardAction,
    legal_actions,
)


def make_config() -> EncounterConfig:
    return EncounterConfig(
        player_max_hp=80,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend", "bash"),
        monsters=(
            MonsterConfig(
                monster_id="jaw_worm",
                name="Jaw Worm",
                max_hp=24,
                intents=(
                    MonsterIntent(intent_id="chomp", damage=11),
                    MonsterIntent(intent_id="thrash", damage=7, block=5),
                ),
            ),
        ),
    )


def test_reset_is_deterministic_for_same_seed() -> None:
    env_a = CombatEnvironment(make_config())
    env_b = CombatEnvironment(make_config())

    state_a = env_a.reset(seed=13)
    state_b = env_b.reset(seed=13)

    assert state_a == state_b


def test_invalid_action_is_rejected_when_target_is_missing() -> None:
    env = CombatEnvironment(
        EncounterConfig(
            player_max_hp=80,
            starting_energy=3,
            draw_per_turn=1,
            deck=("strike",),
            monsters=make_config().monsters,
        )
    )
    env.reset(seed=1)

    try:
        env.step(PlayCardAction(card_index=0))
    except InvalidActionError as exc:
        assert "illegal action" in str(exc)
    else:
        raise AssertionError("expected illegal action to raise InvalidActionError")


def test_legal_actions_are_card_centric_and_target_aware() -> None:
    env = CombatEnvironment(make_config())
    state = env.reset(seed=13)

    actions = legal_actions(state)

    assert EndTurnAction() in actions
    targeted_actions = [action for action in actions if isinstance(action, PlayCardAction)]
    assert targeted_actions
    assert all(action.card_index >= 0 for action in targeted_actions)
    assert all(
        action.target_index == 0 or action.target_index is None for action in targeted_actions
    )


def test_end_turn_applies_enemy_intent_and_rotates_next_intent() -> None:
    env = CombatEnvironment(make_config())
    env.reset(seed=13)

    result = env.step(EndTurnAction())

    assert result.state.turn == 2
    assert result.state.player.hp == 69
    assert result.state.player.energy == 3
    assert result.state.player.block == 0
    assert result.state.monsters[0].intent.intent_id == "thrash"


def test_short_rollout_completes_deterministically() -> None:
    env_a = CombatEnvironment(make_config())
    env_b = CombatEnvironment(make_config())
    env_a.reset(seed=5)
    env_b.reset(seed=5)

    rewards_a: list[float] = []
    rewards_b: list[float] = []
    done = False

    for _ in range(10):
        action_a = choose_damage_first_action(env_a)
        action_b = choose_damage_first_action(env_b)

        assert action_a == action_b

        result_a = env_a.step(action_a)
        result_b = env_b.step(action_b)
        rewards_a.append(result_a.reward)
        rewards_b.append(result_b.reward)

        assert result_a.state == result_b.state
        done = result_a.done
        if done:
            break

    assert rewards_a == rewards_b
    assert done is True


def choose_damage_first_action(env: CombatEnvironment) -> PlayCardAction | EndTurnAction:
    for action in env.legal_actions():
        if isinstance(action, PlayCardAction):
            card = env.state.piles.hand[action.card_index]
            if card.damage > 0:
                return action
    return EndTurnAction()
