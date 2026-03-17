"""Factory helpers for deterministic live-pipeline tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sts_ironclad_rl.integration import GameStateSnapshot
from sts_ironclad_rl.live import BridgeObservationEncoder, EncodedObservation


def make_player(
    *,
    current_hp: int = 70,
    max_hp: int = 80,
    block: int = 0,
    energy: int = 3,
) -> dict[str, int]:
    return {
        "current_hp": current_hp,
        "max_hp": max_hp,
        "block": block,
        "energy": energy,
    }


def make_card(
    *,
    name: str,
    card_id: str | None = None,
    cost: int = 1,
    is_playable: bool = True,
    has_target: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "id": card_id or name.lower().replace(" ", "_"),
        "cost": cost,
        "is_playable": is_playable,
        "has_target": has_target,
    }


def make_enemy(
    *,
    name: str = "Louse",
    monster_id: str | None = None,
    current_hp: int = 18,
    max_hp: int | None = None,
    block: int = 0,
    intent: str | None = None,
    intent_base_damage: int = 0,
    intent_hits: int = 1,
    is_gone: bool = False,
    half_dead: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "id": monster_id or name.lower().replace(" ", "_"),
        "current_hp": current_hp,
        "max_hp": current_hp if max_hp is None else max_hp,
        "block": block,
        "intent": intent,
        "intent_base_damage": intent_base_damage,
        "intent_hits": intent_hits,
        "is_gone": is_gone,
        "half_dead": half_dead,
    }


def make_combat_state(
    *,
    turn: int = 1,
    player: dict[str, Any] | None = None,
    hand: Iterable[dict[str, Any]] = (),
    monsters: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "turn": turn,
        "player": make_player() if player is None else dict(player),
        "hand": [dict(card) for card in hand],
        "monsters": [dict(monster) for monster in monsters],
    }


def make_snapshot(
    *,
    session_id: str = "session-1",
    screen_state: str = "COMBAT",
    available_actions: tuple[str, ...] = ("play", "end"),
    in_combat: bool = True,
    floor: int | None = 3,
    act: int | None = 1,
    raw_state: dict[str, Any] | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        session_id=session_id,
        screen_state=screen_state,
        available_actions=available_actions,
        in_combat=in_combat,
        floor=floor,
        act=act,
        raw_state={} if raw_state is None else dict(raw_state),
    )


def make_combat_snapshot(
    *,
    session_id: str = "session-1",
    available_actions: tuple[str, ...] = ("play", "end"),
    screen_state: str = "COMBAT",
    floor: int | None = 3,
    act: int | None = 1,
    turn: int = 1,
    player: dict[str, Any] | None = None,
    hand: Iterable[dict[str, Any]] = (),
    monsters: Iterable[dict[str, Any]] = (),
    extra_raw_state: dict[str, Any] | None = None,
) -> GameStateSnapshot:
    raw_state = {
        "combat_state": make_combat_state(
            turn=turn,
            player=player,
            hand=hand,
            monsters=monsters,
        )
    }
    if extra_raw_state is not None:
        raw_state.update(extra_raw_state)
    return make_snapshot(
        session_id=session_id,
        screen_state=screen_state,
        available_actions=available_actions,
        in_combat=True,
        floor=floor,
        act=act,
        raw_state=raw_state,
    )


def make_event_snapshot(
    *,
    session_id: str = "session-1",
    available_actions: tuple[str, ...] = ("choose", "proceed", "leave"),
    screen_state: str = "EVENT",
    choice_list: Iterable[str] = ("take", "skip"),
    floor: int | None = 3,
    act: int | None = 1,
    extra_raw_state: dict[str, Any] | None = None,
) -> GameStateSnapshot:
    raw_state: dict[str, Any] = {"choice_list": list(choice_list)}
    if extra_raw_state is not None:
        raw_state.update(extra_raw_state)
    return make_snapshot(
        session_id=session_id,
        screen_state=screen_state,
        available_actions=available_actions,
        in_combat=False,
        floor=floor,
        act=act,
        raw_state=raw_state,
    )


def encode_snapshot(
    snapshot: GameStateSnapshot,
    *,
    encoder: BridgeObservationEncoder | None = None,
) -> EncodedObservation:
    return (BridgeObservationEncoder() if encoder is None else encoder).encode(snapshot)
