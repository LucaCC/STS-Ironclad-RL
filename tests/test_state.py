from dataclasses import replace

from sts_ironclad_rl import (
    EncounterConfig,
    MonsterConfig,
    MonsterIntent,
    PlayerState,
    create_initial_combat_state,
    draw_cards,
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
                max_hp=42,
                intents=(
                    MonsterIntent(intent_id="chomp", damage=11),
                    MonsterIntent(intent_id="thrash", damage=7, block=5),
                ),
            ),
        ),
    )


def test_initial_combat_state_is_reproducible_for_same_seed() -> None:
    state_a = create_initial_combat_state(seed=7, config=make_config())
    state_b = create_initial_combat_state(seed=7, config=make_config())

    assert state_a == state_b
    assert state_a.turn == 1
    assert state_a.player.energy == 3
    assert len(state_a.piles.hand) == 5
    assert state_a.piles.total_cards() == len(make_config().deck)
    assert state_a.monsters[0].intent.intent_id == "chomp"


def test_initial_combat_state_changes_with_seed() -> None:
    state_a = create_initial_combat_state(seed=7, config=make_config())
    state_b = create_initial_combat_state(seed=11, config=make_config())

    assert tuple(card.instance_id for card in state_a.piles.hand) != tuple(
        card.instance_id for card in state_b.piles.hand
    )


def test_draw_cards_reshuffles_discard_when_draw_pile_is_empty() -> None:
    state = create_initial_combat_state(
        seed=3,
        config=EncounterConfig(
            player_max_hp=80,
            starting_energy=3,
            draw_per_turn=2,
            deck=("strike", "defend", "bash", "defend"),
            monsters=make_config().monsters,
        ),
    )
    state = replace(
        state,
        piles=replace(
            state.piles,
            draw_pile=(),
            hand=state.piles.hand,
            discard_pile=state.piles.discard_pile + state.piles.draw_pile,
            exhaust_pile=state.piles.exhaust_pile,
        ),
    )

    result = draw_cards(state, 2)

    assert len(result.drawn_cards) == 2
    assert result.state.piles.total_cards() == state.piles.total_cards()
    assert result.state.shuffle_count == state.shuffle_count + 1


def test_player_state_rejects_invalid_hp() -> None:
    try:
        PlayerState(hp=81, max_hp=80)
    except ValueError as exc:
        assert "hp cannot exceed max_hp" in str(exc)
    else:
        raise AssertionError("expected invalid hp to raise ValueError")
