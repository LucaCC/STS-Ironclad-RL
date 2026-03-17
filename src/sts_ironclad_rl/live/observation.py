"""Stable live-game observation encoding for policy-facing consumers."""

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

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "card_id": self.card_id,
            "name": self.name,
            "cost": self.cost,
            "is_playable": self.is_playable,
            "has_target": self.has_target,
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
    enemy_count: int = 0
    targetable_enemy_count: int = 0
    enemies: tuple[EnemyObservation, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "player": self.player.as_dict(),
            "hand": tuple(card.as_dict() for card in self.hand),
            "enemy_count": self.enemy_count,
            "targetable_enemy_count": self.targetable_enemy_count,
            "enemies": tuple(enemy.as_dict() for enemy in self.enemies),
        }


@dataclass(frozen=True)
class LiveObservation:
    """Typed observation contract normalized from a bridge snapshot."""

    schema_version: str
    screen_state: str
    in_combat: bool
    floor: int | None
    act: int | None
    choice_list: tuple[str, ...]
    available_actions: tuple[str, ...]
    combat: CombatObservation | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "screen_state": self.screen_state,
            "in_combat": self.in_combat,
            "floor": self.floor,
            "act": self.act,
            "choice_list": self.choice_list,
            "available_actions": self.available_actions,
            "combat": None if self.combat is None else self.combat.as_dict(),
        }


@dataclass(frozen=True)
class BridgeObservationEncoder:
    """Encode bridge snapshots into a stable observation contract."""

    layout: ObservationLayout = ObservationLayout()
    action_contract: CommunicationModActionContract = field(
        default_factory=CommunicationModActionContract
    )

    def parse(self, snapshot: GameStateSnapshot) -> LiveObservation:
        raw_state = snapshot.raw_state if isinstance(snapshot.raw_state, dict) else {}
        combat_state = raw_state.get("combat_state")
        combat = _parse_combat_state(combat_state) if isinstance(combat_state, Mapping) else None
        choice_list = tuple(
            choice
            for choice in _iter_sequence(raw_state.get("choice_list"))
            if isinstance(choice, str)
        )
        return LiveObservation(
            schema_version=SCHEMA_VERSION,
            screen_state=snapshot.screen_state,
            in_combat=snapshot.in_combat,
            floor=snapshot.floor,
            act=snapshot.act,
            choice_list=choice_list,
            available_actions=snapshot.available_actions,
            combat=combat,
        )

    def encode(self, snapshot: GameStateSnapshot) -> EncodedObservation:
        observation = self.parse(snapshot)
        combat = observation.combat
        features: dict[str, Any] = {
            "schema_version": observation.schema_version,
            "screen_state": observation.screen_state,
            "in_combat": observation.in_combat,
            "floor": observation.floor if observation.floor is not None else -1,
            "act": observation.act if observation.act is not None else -1,
            "choice_count": len(observation.choice_list),
            "legal_action_count": len(snapshot.available_actions),
            "combat_turn": 0 if combat is None else combat.turn,
            "player_current_hp": 0 if combat is None else combat.player.current_hp,
            "player_max_hp": 0 if combat is None else combat.player.max_hp,
            "player_block": 0 if combat is None else combat.player.block,
            "player_energy": 0 if combat is None else combat.player.energy,
            "hand_count": 0 if combat is None else len(combat.hand),
            "enemy_count": 0 if combat is None else combat.enemy_count,
            "targetable_enemy_count": 0 if combat is None else combat.targetable_enemy_count,
        }
        return EncodedObservation(
            snapshot=snapshot,
            legal_action_ids=self.action_contract.legal_action_ids(snapshot),
            features=features,
            metadata={"structured_observation": observation.as_dict()},
        )


def _parse_combat_state(raw_combat_state: Mapping[str, Any]) -> CombatObservation:
    player = _parse_player(raw_combat_state.get("player"))
    hand = _parse_hand(raw_combat_state.get("hand"))
    enemies = _parse_enemies(raw_combat_state.get("monsters"))
    targetable_enemy_count = sum(1 for enemy in enemies if enemy.is_targetable)
    return CombatObservation(
        turn=_coerce_int(raw_combat_state.get("turn")),
        player=player,
        hand=hand,
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
    for index, raw_card in enumerate(_iter_sequence(raw_hand)):
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
        enemies.append(
            EnemyObservation(
                index=index,
                name=_optional_str(raw_enemy.get("name")),
                monster_id=_optional_str(raw_enemy.get("id")),
                current_hp=current_hp,
                max_hp=_coerce_int(raw_enemy.get("max_hp")),
                block=_coerce_int(raw_enemy.get("block")),
                intent=_optional_str(raw_enemy.get("intent")),
                intent_base_damage=_coerce_int(raw_enemy.get("intent_base_damage")),
                intent_hits=_coerce_int(raw_enemy.get("intent_hits"), default=1),
                is_gone=is_gone,
                half_dead=half_dead,
                is_targetable=not is_gone and not half_dead and current_hp > 0,
            )
        )
    return tuple(enemies)


def _iter_sequence(value: Any) -> Iterable[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _coerce_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return default


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool | int) else False
