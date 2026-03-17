from sts_ironclad_rl import (
    Action,
    CombatantState,
    CombatEnvironment,
    CombatState,
    EncounterConfig,
    HeuristicPolicy,
    PileState,
    RandomPolicy,
)


def make_config() -> EncounterConfig:
    return EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=20,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend"),
    )


def make_state(
    *,
    seed: int = 0,
    turn: int = 1,
    energy: int = 3,
    player_hp: int = 80,
    player_block: int = 0,
    enemy_hp: int = 20,
    hand: tuple[str, ...] = ("strike", "defend"),
) -> CombatState:
    return CombatState(
        seed=seed,
        turn=turn,
        energy=energy,
        starting_energy=3,
        draw_per_turn=5,
        player=CombatantState(hp=player_hp, max_hp=80, block=player_block),
        enemy=CombatantState(hp=enemy_hp, max_hp=20),
        piles=PileState(draw_pile=(), hand=hand, discard_pile=(), exhaust_pile=()),
    )


def test_random_policy_only_chooses_legal_actions() -> None:
    env = CombatEnvironment(make_config())
    state = env.reset(seed=11)
    policy = RandomPolicy()

    for _ in range(20):
        action_mask = env.action_mask()
        action = policy.choose_action(state, action_mask)
        assert action_mask[action] is True


def test_heuristic_policy_only_chooses_legal_actions() -> None:
    env = CombatEnvironment(make_config())
    state = env.reset(seed=5)
    policy = HeuristicPolicy()

    done = False
    steps = 0
    while not done and steps < 10:
        action_mask = env.action_mask()
        action = policy.choose_action(state, action_mask)
        assert action_mask[action] is True
        result = env.step(action)
        state = result.state
        done = result.done
        steps += 1


def test_heuristic_prefers_lethal_attack() -> None:
    state = make_state(enemy_hp=6, hand=("strike", "defend"))

    action = HeuristicPolicy().choose_action(state, _action_mask_for_state(state))

    assert action is Action.ATTACK


def test_heuristic_blocks_when_incoming_damage_is_dangerous() -> None:
    state = make_state(seed=0, turn=1, player_block=0, enemy_hp=20, hand=("strike", "defend"))

    action = HeuristicPolicy().choose_action(state, _action_mask_for_state(state))

    assert action is Action.DEFEND


def test_heuristic_attacks_when_safe() -> None:
    state = make_state(seed=0, turn=1, player_block=7, enemy_hp=20, hand=("strike", "defend"))

    action = HeuristicPolicy().choose_action(state, _action_mask_for_state(state))

    assert action is Action.ATTACK


def test_heuristic_ends_turn_when_no_playable_cards_remain() -> None:
    state = make_state(energy=0, hand=("strike", "defend"))

    action = HeuristicPolicy().choose_action(state, _action_mask_for_state(state))

    assert action is Action.END_TURN


def _action_mask_for_state(state: CombatState) -> dict[Action, bool]:
    return {
        Action.ATTACK: state.energy > 0 and "strike" in state.piles.hand,
        Action.DEFEND: state.energy > 0 and "defend" in state.piles.hand,
        Action.END_TURN: True,
    }
