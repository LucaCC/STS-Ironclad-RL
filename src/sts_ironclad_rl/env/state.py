"""Deterministic combat state primitives for the local battle environment."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

CardId = str


@dataclass(frozen=True)
class CardDefinition:
    """Static card data used to build deterministic card instances."""

    card_id: CardId
    name: str
    cost: int
    damage: int = 0
    block: int = 0
    requires_target: bool = False
    exhausts: bool = False

    def __post_init__(self) -> None:
        if self.cost < 0:
            msg = "cost cannot be negative"
            raise ValueError(msg)
        if self.damage < 0:
            msg = "damage cannot be negative"
            raise ValueError(msg)
        if self.block < 0:
            msg = "block cannot be negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class CardState:
    """Concrete card instance tracked through draw, hand, discard, and exhaust piles."""

    instance_id: str
    card_id: CardId
    name: str
    cost: int
    damage: int = 0
    block: int = 0
    requires_target: bool = False
    exhausts: bool = False


@dataclass(frozen=True)
class MonsterIntent:
    """Deterministic monster move description for the next enemy turn."""

    intent_id: str
    damage: int = 0
    hits: int = 1
    block: int = 0

    def __post_init__(self) -> None:
        if self.damage < 0:
            msg = "damage cannot be negative"
            raise ValueError(msg)
        if self.hits <= 0:
            msg = "hits must be positive"
            raise ValueError(msg)
        if self.block < 0:
            msg = "block cannot be negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class MonsterConfig:
    """Encounter-time monster definition with a deterministic intent cycle."""

    monster_id: str
    name: str
    max_hp: int
    intents: tuple[MonsterIntent, ...]

    def __post_init__(self) -> None:
        if self.max_hp <= 0:
            msg = "max_hp must be positive"
            raise ValueError(msg)
        if not self.intents:
            msg = "monsters must define at least one intent"
            raise ValueError(msg)


@dataclass(frozen=True)
class PlayerState:
    """Player combat state used by the deterministic combat simulator."""

    hp: int
    max_hp: int
    block: int = 0
    energy: int = 0

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
        if self.energy < 0:
            msg = "energy cannot be negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class MonsterState:
    """Monster combat state with a deterministic repeating intent cycle."""

    monster_id: str
    name: str
    hp: int
    max_hp: int
    block: int = 0
    intent_cycle: tuple[MonsterIntent, ...] = ()
    intent_index: int = 0

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
        if not self.intent_cycle:
            msg = "intent_cycle cannot be empty"
            raise ValueError(msg)
        if not 0 <= self.intent_index < len(self.intent_cycle):
            msg = "intent_index must reference the current intent cycle"
            raise ValueError(msg)

    @property
    def intent(self) -> MonsterIntent:
        """Return the current intent from the monster's deterministic cycle."""
        return self.intent_cycle[self.intent_index]

    @property
    def is_alive(self) -> bool:
        """Return whether the monster can still act and be targeted."""
        return self.hp > 0


@dataclass(frozen=True)
class PileState:
    """Card zones required for deterministic setup and draw simulation."""

    draw_pile: tuple[CardState, ...]
    hand: tuple[CardState, ...] = ()
    discard_pile: tuple[CardState, ...] = ()
    exhaust_pile: tuple[CardState, ...] = ()

    def total_cards(self) -> int:
        """Return the number of cards tracked across all zones."""
        return (
            len(self.draw_pile) + len(self.hand) + len(self.discard_pile) + len(self.exhaust_pile)
        )


@dataclass(frozen=True)
class EncounterConfig:
    """Inputs required to create an initial deterministic combat state."""

    player_max_hp: int
    starting_energy: int
    draw_per_turn: int
    deck: tuple[CardId, ...]
    monsters: tuple[MonsterConfig, ...]

    def __post_init__(self) -> None:
        if self.player_max_hp <= 0:
            msg = "player_max_hp must be positive"
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
        if not self.monsters:
            msg = "encounters must include at least one monster"
            raise ValueError(msg)
        unknown_cards = tuple(card_id for card_id in self.deck if card_id not in CARD_LIBRARY)
        if unknown_cards:
            msg = f"unknown card ids: {', '.join(unknown_cards)}"
            raise ValueError(msg)


@dataclass(frozen=True)
class CombatState:
    """Deterministic combat snapshot for a single encounter."""

    seed: int
    turn: int
    starting_energy: int
    draw_per_turn: int
    shuffle_count: int
    player: PlayerState
    monsters: tuple[MonsterState, ...]
    piles: PileState

    def is_terminal(self) -> bool:
        """Return whether the combat has reached a terminal win/loss state."""
        return self.player.hp == 0 or not any(monster.is_alive for monster in self.monsters)


@dataclass(frozen=True)
class DrawResult:
    """Result of drawing cards from an existing combat state."""

    state: CombatState
    drawn_cards: tuple[CardState, ...]


CARD_LIBRARY: dict[CardId, CardDefinition] = {
    "strike": CardDefinition(
        card_id="strike",
        name="Strike",
        cost=1,
        damage=6,
        requires_target=True,
    ),
    "defend": CardDefinition(
        card_id="defend",
        name="Defend",
        cost=1,
        block=5,
    ),
    "bash": CardDefinition(
        card_id="bash",
        name="Bash",
        cost=2,
        damage=8,
        requires_target=True,
    ),
}


def create_initial_combat_state(seed: int, config: EncounterConfig) -> CombatState:
    """Create a reproducible opening combat state from a seed and encounter config."""
    deck = _build_deck_instances(config.deck)
    shuffled_deck = _shuffle_cards(deck, seed=seed, shuffle_count=0)
    monsters = tuple(
        MonsterState(
            monster_id=monster.monster_id,
            name=monster.name,
            hp=monster.max_hp,
            max_hp=monster.max_hp,
            block=0,
            intent_cycle=monster.intents,
            intent_index=0,
        )
        for monster in config.monsters
    )
    initial_state = CombatState(
        seed=seed,
        turn=1,
        starting_energy=config.starting_energy,
        draw_per_turn=config.draw_per_turn,
        shuffle_count=1,
        player=PlayerState(
            hp=config.player_max_hp,
            max_hp=config.player_max_hp,
            block=0,
            energy=config.starting_energy,
        ),
        monsters=monsters,
        piles=PileState(draw_pile=shuffled_deck),
    )
    return draw_cards(initial_state, config.draw_per_turn).state


def draw_cards(state: CombatState, count: int) -> DrawResult:
    """Move up to ``count`` cards into the hand, reshuffling discard when needed."""
    if count < 0:
        msg = "count cannot be negative"
        raise ValueError(msg)

    next_state = state
    drawn_cards: list[CardState] = []
    draw_pile = list(state.piles.draw_pile)
    discard_pile = list(state.piles.discard_pile)
    hand = list(state.piles.hand)
    shuffle_count = state.shuffle_count

    while len(drawn_cards) < count:
        if not draw_pile:
            if not discard_pile:
                break
            draw_pile = list(
                _shuffle_cards(
                    tuple(discard_pile),
                    seed=state.seed,
                    shuffle_count=shuffle_count,
                )
            )
            discard_pile = []
            shuffle_count += 1

        next_card = draw_pile.pop(0)
        drawn_cards.append(next_card)
        hand.append(next_card)

    next_state = CombatState(
        seed=state.seed,
        turn=state.turn,
        starting_energy=state.starting_energy,
        draw_per_turn=state.draw_per_turn,
        shuffle_count=shuffle_count,
        player=state.player,
        monsters=state.monsters,
        piles=PileState(
            draw_pile=tuple(draw_pile),
            hand=tuple(hand),
            discard_pile=tuple(discard_pile),
            exhaust_pile=state.piles.exhaust_pile,
        ),
    )
    return DrawResult(state=next_state, drawn_cards=tuple(drawn_cards))


def _build_deck_instances(deck: tuple[CardId, ...]) -> tuple[CardState, ...]:
    cards: list[CardState] = []
    counts: dict[CardId, int] = {}
    for card_id in deck:
        definition = CARD_LIBRARY[card_id]
        counts[card_id] = counts.get(card_id, 0) + 1
        cards.append(
            CardState(
                instance_id=f"{card_id}-{counts[card_id]}",
                card_id=definition.card_id,
                name=definition.name,
                cost=definition.cost,
                damage=definition.damage,
                block=definition.block,
                requires_target=definition.requires_target,
                exhausts=definition.exhausts,
            )
        )
    return tuple(cards)


def _shuffle_cards(
    cards: tuple[CardState, ...],
    *,
    seed: int,
    shuffle_count: int,
) -> tuple[CardState, ...]:
    shuffled_cards = list(cards)
    Random(seed + (shuffle_count * 9973)).shuffle(shuffled_cards)
    return tuple(shuffled_cards)
