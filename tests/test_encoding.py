from sts_ironclad_rl import (
    ACTION_ORDER,
    OBSERVATION_FIELDS,
    Action,
    CombatEnvironment,
    EncounterConfig,
    InvalidActionError,
    action_to_index,
    create_initial_combat_state,
    decode_action_index,
    encode_action_mask,
    encode_observation,
    legal_action_mask,
)


def make_config() -> EncounterConfig:
    return EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=20,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend"),
    )


def test_action_indices_are_stable_and_round_trip() -> None:
    assert ACTION_ORDER == (Action.ATTACK, Action.DEFEND, Action.END_TURN)
    assert action_to_index(Action.ATTACK) == 0
    assert action_to_index(Action.DEFEND) == 1
    assert action_to_index(Action.END_TURN) == 2
    assert decode_action_index(0) is Action.ATTACK
    assert decode_action_index(1) is Action.DEFEND
    assert decode_action_index(2) is Action.END_TURN


def test_action_mask_encoding_and_illegal_actions_are_explicit() -> None:
    env = CombatEnvironment(
        EncounterConfig(
            player_max_hp=80,
            enemy_max_hp=20,
            starting_energy=1,
            draw_per_turn=1,
            deck=("strike",),
        )
    )
    env.reset(seed=7)

    assert env.action_mask() == {
        Action.ATTACK: True,
        Action.DEFEND: False,
        Action.END_TURN: True,
    }
    assert env.legal_action_mask() == (True, False, True)
    assert encode_action_mask(env.action_mask()) == (True, False, True)

    try:
        env.step_index(action_to_index(Action.DEFEND))
    except InvalidActionError as exc:
        assert "illegal action" in str(exc)
    else:
        raise AssertionError("expected illegal action to raise InvalidActionError")


def test_state_level_legal_action_mask_tracks_energy_and_hand() -> None:
    state = create_initial_combat_state(
        seed=3,
        config=EncounterConfig(
            player_max_hp=80,
            enemy_max_hp=20,
            starting_energy=0,
            draw_per_turn=1,
            deck=("defend",),
        ),
    )

    assert legal_action_mask(state) == (False, False, True)


def test_observation_schema_is_stable_and_deterministic_for_seeded_reset() -> None:
    env_a = CombatEnvironment(make_config())
    env_b = CombatEnvironment(make_config())

    state_a = env_a.reset(seed=13)
    env_b.reset(seed=13)

    observation_a = env_a.observation()
    observation_b = env_b.observation()

    assert OBSERVATION_FIELDS == (
        "player_hp",
        "player_max_hp",
        "player_block",
        "energy",
        "turn",
        "enemy_hp",
        "enemy_max_hp",
        "enemy_block",
        "enemy_intent_damage",
        "hand_strike_count",
        "hand_defend_count",
        "draw_strike_count",
        "draw_defend_count",
        "discard_strike_count",
        "discard_defend_count",
        "draw_pile_count",
        "hand_count",
        "discard_pile_count",
        "exhaust_pile_count",
    )
    assert len(observation_a) == len(OBSERVATION_FIELDS)
    assert observation_a == observation_b
    assert observation_a == (80, 80, 0, 3, 1, 20, 20, 0, 6, 3, 2, 0, 0, 0, 0, 0, 5, 0, 0)
    assert encode_observation(state_a) == observation_a


def test_observation_content_is_deterministic_after_fixed_action_sequence() -> None:
    env = CombatEnvironment(make_config())
    env.reset(seed=13)

    env.step_index(action_to_index(Action.DEFEND))
    env.step_index(action_to_index(Action.END_TURN))

    assert env.observation() == (79, 80, 0, 3, 2, 20, 20, 0, 7, 3, 2, 0, 0, 0, 0, 0, 5, 0, 0)
