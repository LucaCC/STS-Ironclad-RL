from sts_ironclad_rl import EncounterConfig, create_initial_combat_state, draw_cards
from sts_ironclad_rl.env.state import CombatantState


def test_initial_combat_state_is_reproducible_for_same_seed() -> None:
    config = EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=40,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend", "bash"),
    )

    state_a = create_initial_combat_state(seed=7, config=config)
    state_b = create_initial_combat_state(seed=7, config=config)

    assert state_a == state_b
    assert state_a.turn == 1
    assert state_a.energy == 3
    assert len(state_a.piles.hand) == 5
    assert state_a.piles.total_cards() == len(config.deck)


def test_initial_combat_state_changes_with_seed() -> None:
    config = EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=40,
        starting_energy=3,
        draw_per_turn=5,
        deck=("strike", "strike", "strike", "defend", "defend", "bash"),
    )

    state_a = create_initial_combat_state(seed=7, config=config)
    state_b = create_initial_combat_state(seed=11, config=config)

    assert state_a.piles.hand != state_b.piles.hand


def test_draw_cards_preserves_card_count_and_order() -> None:
    config = EncounterConfig(
        player_max_hp=80,
        enemy_max_hp=40,
        starting_energy=3,
        draw_per_turn=2,
        deck=("strike", "defend", "bash", "defend"),
    )
    state = create_initial_combat_state(seed=3, config=config)

    result = draw_cards(state, 2)

    assert result.drawn_cards == state.piles.draw_pile[:2]
    assert result.state.piles.total_cards() == state.piles.total_cards()
    assert result.state.piles.hand == state.piles.hand + result.drawn_cards


def test_combatant_state_rejects_invalid_hp() -> None:
    try:
        CombatantState(hp=81, max_hp=80)
    except ValueError as exc:
        assert "hp cannot exceed max_hp" in str(exc)
    else:
        raise AssertionError("expected invalid hp to raise ValueError")
