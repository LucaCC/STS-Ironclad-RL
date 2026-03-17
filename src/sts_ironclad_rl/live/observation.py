"""Stable live-game observation encoding for RL-facing consumers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..integration import GameStateSnapshot
from .actions import CommunicationModActionContract
from .contracts import EncodedObservation

SCHEMA_VERSION = "live_observation.v1"


@dataclass(frozen=True)
class ObservationLayout:
    """Fixed-size export layout for dict and vector-friendly observations."""

    max_hand_cards: int = 10
    max_enemies: int = 5

    def vector_length(self) -> int:
        """Return the number of numeric features emitted by the encoder."""
        return len(vector_schema(self))


@dataclass(frozen=True)
class PlayerObservation:
    """Normalized player state from a live combat snapshot."""

    current_hp: int = 0
    max_hp: int = 0
    block: int = 0
    energy: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "block": self.block,
            "energy": self.energy,
        }


@dataclass(frozen=True)
class CardObservation:
    """One normalized hand-card slot in bridge order."""

    index: int
    card_id: str | None = None
    name: str | None = None
    cost: int = -1
    is_playable: bool = False
    has_target: bool = False
    upgraded: bool = False
    exhausts: bool = False
    ethereal: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "card_id": self.card_id,
            "name": self.name,
            "cost": self.cost,
            "is_playable": self.is_playable,
            "has_target": self.has_target,
            "upgraded": self.upgraded,
            "exhausts": self.exhausts,
            "ethereal": self.ethereal,
        }


@dataclass(frozen=True)
class EnemyObservation:
    """One normalized enemy slot in bridge order."""

    index: int
    name: str | None = None
    monster_id: str | None = None
    current_hp: int = 0
    max_hp: int = 0
    block: int = 0
    intent: str | None = None
    move_id: int | None = None
    intent_base_damage: int = 0
    intent_hits: int = 0
    is_gone: bool = False
    half_dead: bool = False
    is_targetable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "id": self.monster_id,
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "block": self.block,
            "intent": self.intent,
            "move_id": self.move_id,
            "intent_base_damage": self.intent_base_damage,
            "intent_hits": self.intent_hits,
            "is_gone": self.is_gone,
            "half_dead": self.half_dead,
            "is_targetable": self.is_targetable,
        }


@dataclass(frozen=True)
class CombatObservation:
    """Combat-specific normalized state."""

    turn: int = 0
    player: PlayerObservation = PlayerObservation()
    hand: tuple[CardObservation, ...] = ()
    draw_pile_size: int = 0
    discard_pile_size: int = 0
    exhaust_pile_size: int = 0
    enemy_count: int = 0
    targetable_enemy_count: int = 0
    enemies: tuple[EnemyObservation, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "player": self.player.as_dict(),
            "hand": tuple(card.as_dict() for card in self.hand),
            "draw_pile_size": self.draw_pile_size,
            "discard_pile_size": self.discard_pile_size,
            "exhaust_pile_size": self.exhaust_pile_size,
            "enemy_count": self.enemy_count,
            "targetable_enemy_count": self.targetable_enemy_count,
            "enemies": tuple(enemy.as_dict() for enemy in self.enemies),
        }


@dataclass(frozen=True)
class LiveObservation:
    """Typed observation contract normalized from a bridge snapshot."""

    schema_version: str
    screen_state: str
    screen_type: str | None
    room_phase: str | None
    action_phase: str | None
    in_combat: bool
    floor: int | None
    act: int | None
    ascension_level: int | None
    gold: int | None
    class_name: str | None
    choice_list: tuple[str, ...]
    available_actions: tuple[str, ...]
    combat: CombatObservation | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a structured, deterministic dictionary form."""
        return {
            "schema_version": self.schema_version,
            "screen_state": self.screen_state,
            "screen_type": self.screen_type,
            "room_phase": self.room_phase,
            "action_phase": self.action_phase,
            "in_combat": self.in_combat,
            "floor": self.floor,
            "act": self.act,
            "ascension_level": self.ascension_level,
            "gold": self.gold,
            "class_name": self.class_name,
            "choice_list": self.choice_list,
            "available_actions": self.available_actions,
            "combat": None if self.combat is None else self.combat.as_dict(),
        }

    def flat_dict(self, layout: ObservationLayout) -> dict[str, int | float | bool | str]:
        """Return a stable flat feature dictionary with padded slot keys."""
        return flatten_observation(self, layout)

    def vector(self, layout: ObservationLayout) -> tuple[float, ...]:
        """Return the numeric vector export aligned with ``vector_schema``."""
        feature_map = self.flat_dict(layout)
        return tuple(float(feature_map[name]) for name in vector_schema(layout))


@dataclass(frozen=True)
class BridgeObservationEncoder:
    """Encode loose live-game bridge snapshots into a stable observation contract."""

    layout: ObservationLayout = ObservationLayout()
    action_contract: CommunicationModActionContract = field(
        default_factory=CommunicationModActionContract
    )

    def parse(self, snapshot: GameStateSnapshot) -> LiveObservation:
        """Parse raw bridge state into typed observation dataclasses."""
        raw_state = snapshot.raw_state if isinstance(snapshot.raw_state, dict) else {}
        combat_state = raw_state.get("combat_state")
        combat = _parse_combat_state(combat_state) if isinstance(combat_state, dict) else None
        choice_list = tuple(
            choice
            for choice in _iter_sequence(raw_state.get("choice_list"))
            if isinstance(choice, str)
        )
        return LiveObservation(
            schema_version=SCHEMA_VERSION,
            screen_state=snapshot.screen_state,
            screen_type=_optional_str(raw_state.get("screen_type")),
            room_phase=_optional_str(raw_state.get("room_phase")),
            action_phase=_optional_str(raw_state.get("action_phase")),
            in_combat=snapshot.in_combat,
            floor=snapshot.floor,
            act=snapshot.act,
            ascension_level=_optional_int(raw_state.get("ascension_level")),
            gold=_optional_int(raw_state.get("gold")),
            class_name=_optional_str(raw_state.get("class")),
            choice_list=choice_list,
            available_actions=snapshot.available_actions,
            combat=combat,
        )

    def encode(self, snapshot: GameStateSnapshot) -> EncodedObservation:
        """Encode a snapshot into the repo-level observation contract."""
        observation = self.parse(snapshot)
        legal_action_ids = self.action_contract.legal_action_ids(snapshot)
        return EncodedObservation(
            snapshot=snapshot,
            legal_action_ids=legal_action_ids,
            features=observation.flat_dict(self.layout),
            metadata={
                "schema_version": observation.schema_version,
                "structured_observation": observation.as_dict(),
                "available_actions": observation.available_actions,
                "vector_schema": vector_schema(self.layout),
                "vector_length": self.layout.vector_length(),
            },
        )


def flatten_observation(
    observation: LiveObservation,
    layout: ObservationLayout,
) -> dict[str, int | float | bool | str]:
    """Return a stable, flat mapping suitable for simple policy adapters."""
    combat = observation.combat
    hand = combat.hand if combat is not None else ()
    enemies = combat.enemies if combat is not None else ()
    features: dict[str, int | float | bool | str] = {
        "schema_version": observation.schema_version,
        "screen_state": observation.screen_state,
        "screen_type": observation.screen_type or "",
        "room_phase": observation.room_phase or "",
        "action_phase": observation.action_phase or "",
        "in_combat": observation.in_combat,
        "floor": observation.floor if observation.floor is not None else -1,
        "act": observation.act if observation.act is not None else -1,
        "ascension_level": observation.ascension_level
        if observation.ascension_level is not None
        else -1,
        "gold": observation.gold if observation.gold is not None else -1,
        "choice_count": len(observation.choice_list),
        "legal_action_count": len(observation.available_actions),
        "combat_present": combat is not None,
        "combat_turn": combat.turn if combat is not None else 0,
        "player_current_hp": combat.player.current_hp if combat is not None else 0,
        "player_max_hp": combat.player.max_hp if combat is not None else 0,
        "player_block": combat.player.block if combat is not None else 0,
        "player_energy": combat.player.energy if combat is not None else 0,
        "draw_pile_size": combat.draw_pile_size if combat is not None else 0,
        "discard_pile_size": combat.discard_pile_size if combat is not None else 0,
        "exhaust_pile_size": combat.exhaust_pile_size if combat is not None else 0,
        "enemy_count": combat.enemy_count if combat is not None else 0,
        "targetable_enemy_count": combat.targetable_enemy_count if combat is not None else 0,
        "hand_count": len(hand),
        "playable_card_count": sum(1 for card in hand if card.is_playable),
        "targeted_card_count": sum(1 for card in hand if card.has_target),
    }

    for slot in range(layout.max_hand_cards):
        prefix = f"hand_{slot}"
        if slot < len(hand):
            card = hand[slot]
            features[f"{prefix}_present"] = True
            features[f"{prefix}_cost"] = card.cost
            features[f"{prefix}_is_playable"] = card.is_playable
            features[f"{prefix}_has_target"] = card.has_target
            features[f"{prefix}_upgraded"] = card.upgraded
            features[f"{prefix}_exhausts"] = card.exhausts
            features[f"{prefix}_ethereal"] = card.ethereal
        else:
            features[f"{prefix}_present"] = False
            features[f"{prefix}_cost"] = -1
            features[f"{prefix}_is_playable"] = False
            features[f"{prefix}_has_target"] = False
            features[f"{prefix}_upgraded"] = False
            features[f"{prefix}_exhausts"] = False
            features[f"{prefix}_ethereal"] = False

    for slot in range(layout.max_enemies):
        prefix = f"enemy_{slot}"
        if slot < len(enemies):
            enemy = enemies[slot]
            features[f"{prefix}_present"] = True
            features[f"{prefix}_current_hp"] = enemy.current_hp
            features[f"{prefix}_max_hp"] = enemy.max_hp
            features[f"{prefix}_block"] = enemy.block
            features[f"{prefix}_intent_base_damage"] = enemy.intent_base_damage
            features[f"{prefix}_intent_hits"] = enemy.intent_hits
            features[f"{prefix}_is_gone"] = enemy.is_gone
            features[f"{prefix}_half_dead"] = enemy.half_dead
            features[f"{prefix}_is_targetable"] = enemy.is_targetable
        else:
            features[f"{prefix}_present"] = False
            features[f"{prefix}_current_hp"] = 0
            features[f"{prefix}_max_hp"] = 0
            features[f"{prefix}_block"] = 0
            features[f"{prefix}_intent_base_damage"] = 0
            features[f"{prefix}_intent_hits"] = 0
            features[f"{prefix}_is_gone"] = False
            features[f"{prefix}_half_dead"] = False
            features[f"{prefix}_is_targetable"] = False

    return features


def vector_schema(layout: ObservationLayout) -> tuple[str, ...]:
    """Return the ordered numeric feature names used by ``LiveObservation.vector``."""
    names = [
        "in_combat",
        "floor",
        "act",
        "ascension_level",
        "gold",
        "choice_count",
        "legal_action_count",
        "combat_present",
        "combat_turn",
        "player_current_hp",
        "player_max_hp",
        "player_block",
        "player_energy",
        "draw_pile_size",
        "discard_pile_size",
        "exhaust_pile_size",
        "enemy_count",
        "targetable_enemy_count",
        "hand_count",
        "playable_card_count",
        "targeted_card_count",
    ]
    for slot in range(layout.max_hand_cards):
        prefix = f"hand_{slot}"
        names.extend(
            [
                f"{prefix}_present",
                f"{prefix}_cost",
                f"{prefix}_is_playable",
                f"{prefix}_has_target",
                f"{prefix}_upgraded",
                f"{prefix}_exhausts",
                f"{prefix}_ethereal",
            ]
        )
    for slot in range(layout.max_enemies):
        prefix = f"enemy_{slot}"
        names.extend(
            [
                f"{prefix}_present",
                f"{prefix}_current_hp",
                f"{prefix}_max_hp",
                f"{prefix}_block",
                f"{prefix}_intent_base_damage",
                f"{prefix}_intent_hits",
                f"{prefix}_is_gone",
                f"{prefix}_half_dead",
                f"{prefix}_is_targetable",
            ]
        )
    return tuple(names)


def _parse_combat_state(raw_combat_state: Mapping[str, Any]) -> CombatObservation:
    player = _parse_player(raw_combat_state.get("player"))
    hand = _parse_hand(raw_combat_state.get("hand"))
    enemies = _parse_enemies(raw_combat_state.get("monsters"))
    targetable_enemy_count = sum(1 for enemy in enemies if enemy.is_targetable)
    return CombatObservation(
        turn=_coerce_int(raw_combat_state.get("turn")),
        player=player,
        hand=hand,
        draw_pile_size=_sequence_length(raw_combat_state.get("draw_pile")),
        discard_pile_size=_sequence_length(raw_combat_state.get("discard_pile")),
        exhaust_pile_size=_sequence_length(raw_combat_state.get("exhaust_pile")),
        enemy_count=len(enemies),
        targetable_enemy_count=targetable_enemy_count,
        enemies=enemies,
    )


def _parse_player(raw_player: Any) -> PlayerObservation:
    if not isinstance(raw_player, Mapping):
        return PlayerObservation()
    return PlayerObservation(
        current_hp=_coerce_int(raw_player.get("current_hp")),
        max_hp=_coerce_int(raw_player.get("max_hp")),
        block=_coerce_int(raw_player.get("block")),
        energy=_coerce_int(raw_player.get("energy")),
    )


def _parse_hand(raw_hand: Any) -> tuple[CardObservation, ...]:
    cards: list[CardObservation] = []
    for index, raw_card in enumerate(_iter_sequence(raw_hand), start=1):
        if not isinstance(raw_card, Mapping):
            continue
        cards.append(
            CardObservation(
                index=index,
                card_id=_optional_str(raw_card.get("id")),
                name=_optional_str(raw_card.get("name")),
                cost=_coerce_int(raw_card.get("cost"), default=-1),
                is_playable=_coerce_bool(raw_card.get("is_playable")),
                has_target=_coerce_bool(raw_card.get("has_target")),
                upgraded=_coerce_bool(raw_card.get("upgraded")),
                exhausts=_coerce_bool(raw_card.get("exhausts") or raw_card.get("exhaust")),
                ethereal=_coerce_bool(raw_card.get("ethereal")),
            )
        )
    return tuple(cards)


def _parse_enemies(raw_enemies: Any) -> tuple[EnemyObservation, ...]:
    enemies: list[EnemyObservation] = []
    for index, raw_enemy in enumerate(_iter_sequence(raw_enemies)):
        if not isinstance(raw_enemy, Mapping):
            continue
        current_hp = _coerce_int(raw_enemy.get("current_hp"))
        is_gone = _coerce_bool(raw_enemy.get("is_gone"))
        half_dead = _coerce_bool(raw_enemy.get("half_dead"))
        is_targetable = not is_gone and not half_dead and current_hp > 0
        enemies.append(
            EnemyObservation(
                index=index,
                name=_optional_str(raw_enemy.get("name")),
                monster_id=_optional_str(raw_enemy.get("id")),
                current_hp=current_hp,
                max_hp=_coerce_int(raw_enemy.get("max_hp")),
                block=_coerce_int(raw_enemy.get("block")),
                intent=_optional_str(raw_enemy.get("intent")),
                move_id=_optional_int(raw_enemy.get("move_id")),
                intent_base_damage=_coerce_int(raw_enemy.get("intent_base_damage")),
                intent_hits=_coerce_int(raw_enemy.get("intent_hits")),
                is_gone=is_gone,
                half_dead=half_dead,
                is_targetable=is_targetable,
            )
        )
    return tuple(enemies)


def _iter_sequence(value: Any) -> Iterable[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _sequence_length(value: Any) -> int:
    return sum(1 for _ in _iter_sequence(value))


def _coerce_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return default


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool | int) else False
