"""Deterministic combat state primitives."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

CardId = str


@dataclass(frozen=True)
class CombatantState:
    """Minimal combatant state for combat-only simulation."""

    hp: int
    max_hp: int
    block: int = 0

    def __post_init__(self) -> None:
        if self.max_hp <= 0:
            msg = "max_hp must be positive"
            raise ValueError(msg)
        if self.hp < 0:
            msg = "hp cannot be negative"
            raise ValueError(msg)
        if self.hp > self.max_hp:
            msg = "hp cannot exceed max_hp"
            raise ValueError(msg)
        if self.block < 0:
            msg = "block cannot be negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class PileState:
    """Card zones required for deterministic setup and draw simulation."""

    draw_pile: tuple[CardId, ...]
    hand: tuple[CardId, ...] = ()
    discard_pile: tuple[CardId, ...] = ()
    exhaust_pile: tuple[CardId, ...] = ()

    def total_cards(self) -> int:
        """Return the number of cards tracked across all zones."""
        return (
            len(self.draw_pile) + len(self.hand) + len(self.discard_pile) + len(self.exhaust_pile)
        )


@dataclass(frozen=True)
class EncounterConfig:
    """Inputs required to create an initial combat state."""

    player_max_hp: int
    enemy_max_hp: int
    starting_energy: int
    draw_per_turn: int
    deck: tuple[CardId, ...]

    def __post_init__(self) -> None:
        if self.player_max_hp <= 0:
            msg = "player_max_hp must be positive"
            raise ValueError(msg)
        if self.enemy_max_hp <= 0:
            msg = "enemy_max_hp must be positive"
            raise ValueError(msg)
        if self.starting_energy < 0:
            msg = "starting_energy cannot be negative"
            raise ValueError(msg)
        if self.draw_per_turn <= 0:
            msg = "draw_per_turn must be positive"
            raise ValueError(msg)
        if not self.deck:
            msg = "deck cannot be empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class CombatState:
    """Minimal deterministic combat state for a single encounter."""

    seed: int
    turn: int
    energy: int
    draw_per_turn: int
    player: CombatantState
    enemy: CombatantState
    piles: PileState


@dataclass(frozen=True)
class DrawResult:
    """Result of drawing cards from an existing combat state."""

    state: CombatState
    drawn_cards: tuple[CardId, ...]


def create_initial_combat_state(seed: int, config: EncounterConfig) -> CombatState:
    """Create a reproducible opening combat state from a seed and encounter config."""
    shuffled_deck = _shuffle_deck(config.deck, seed)
    piles = PileState(draw_pile=shuffled_deck)
    initial_state = CombatState(
        seed=seed,
        turn=1,
        energy=config.starting_energy,
        draw_per_turn=config.draw_per_turn,
        player=CombatantState(hp=config.player_max_hp, max_hp=config.player_max_hp),
        enemy=CombatantState(hp=config.enemy_max_hp, max_hp=config.enemy_max_hp),
        piles=piles,
    )
    return draw_cards(initial_state, config.draw_per_turn).state


def draw_cards(state: CombatState, count: int) -> DrawResult:
    """Move up to ``count`` cards from the draw pile into the hand."""
    if count < 0:
        msg = "count cannot be negative"
        raise ValueError(msg)

    drawn_cards = state.piles.draw_pile[:count]
    remaining_draw_pile = state.piles.draw_pile[count:]
    next_hand = state.piles.hand + drawn_cards
    next_piles = PileState(
        draw_pile=remaining_draw_pile,
        hand=next_hand,
        discard_pile=state.piles.discard_pile,
        exhaust_pile=state.piles.exhaust_pile,
    )
    next_state = CombatState(
        seed=state.seed,
        turn=state.turn,
        energy=state.energy,
        draw_per_turn=state.draw_per_turn,
        player=state.player,
        enemy=state.enemy,
        piles=next_piles,
    )
    return DrawResult(state=next_state, drawn_cards=drawn_cards)


def _shuffle_deck(deck: tuple[CardId, ...], seed: int) -> tuple[CardId, ...]:
    cards = list(deck)
    Random(seed).shuffle(cards)
    return tuple(cards)
