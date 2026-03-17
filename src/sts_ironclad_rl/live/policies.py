"""Baseline policies for live-game smoke runs and debugging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

from .actions import (
    ChooseAction,
    EndTurnAction,
    LeaveAction,
    PlayCardAction,
    ProceedAction,
    action_from_id,
)
from .contracts import ActionDecision, EncodedObservation

_DEFENSIVE_CARD_TOKENS = ("defend", "shrug", "flame barrier", "ghostly armor", "impervious")


@dataclass
class RandomLegalPolicy:
    """Uniform random baseline over the observation's legal actions."""

    seed: int | None = None
    name: str = "random_legal"
    _random: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = Random(self.seed)

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        legal_action_ids = tuple(observation.legal_action_ids)
        if not legal_action_ids:
            msg = "observation does not expose any legal actions"
            raise ValueError(msg)
        return ActionDecision(action_id=self._random.choice(legal_action_ids))


@dataclass(frozen=True)
class SimpleHeuristicPolicy:
    """Deterministic, conservative baseline for live combat smoke tests."""

    name: str = "simple_heuristic"

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        legal_action_ids = tuple(observation.legal_action_ids)
        if not legal_action_ids:
            msg = "observation does not expose any legal actions"
            raise ValueError(msg)

        structured = _structured_observation(observation)
        if not structured.get("in_combat", False):
            return ActionDecision(action_id=_best_noncombat_action(legal_action_ids))

        defensive_action = _best_defensive_action(legal_action_ids, structured)
        if defensive_action is not None:
            return ActionDecision(action_id=defensive_action)

        targeted_attack = _best_targeted_attack_action(legal_action_ids, structured)
        if targeted_attack is not None:
            return ActionDecision(action_id=targeted_attack)

        non_targeted = _first_play_card_without_target(legal_action_ids)
        if non_targeted is not None:
            return ActionDecision(action_id=non_targeted)

        if EndTurnAction().action_id in legal_action_ids:
            return ActionDecision(action_id=EndTurnAction().action_id)

        return ActionDecision(action_id=sorted(legal_action_ids)[0])


def _structured_observation(observation: EncodedObservation) -> Mapping[str, Any]:
    structured = observation.metadata.get("structured_observation")
    if isinstance(structured, Mapping):
        return structured
    return {}


def _best_noncombat_action(legal_action_ids: Sequence[str]) -> str:
    for action_id in (
        ProceedAction().action_id,
        ChooseAction(choice_index=0).action_id,
        LeaveAction().action_id,
    ):
        if action_id in legal_action_ids:
            return action_id
    return sorted(legal_action_ids)[0]


def _best_defensive_action(
    legal_action_ids: Sequence[str],
    structured_observation: Mapping[str, Any],
) -> str | None:
    combat = structured_observation.get("combat")
    if not isinstance(combat, Mapping):
        return None

    player = combat.get("player")
    if not isinstance(player, Mapping):
        return None

    current_hp = _as_int(player.get("current_hp"))
    max_hp = _as_int(player.get("max_hp"))
    current_block = _as_int(player.get("block"))
    incoming_damage = _incoming_damage(combat.get("enemies"))

    if max_hp <= 0:
        return None
    if current_hp * 2 > max_hp:
        return None
    if incoming_damage <= current_block:
        return None

    hand = combat.get("hand")
    if not isinstance(hand, Sequence):
        return None

    for action_id in legal_action_ids:
        action = action_from_id(action_id)
        if not isinstance(action, PlayCardAction) or action.target is not None:
            continue
        if action.hand_index >= len(hand):
            continue
        raw_card = hand[action.hand_index]
        if not isinstance(raw_card, Mapping):
            continue
        if _is_defensive_card(raw_card):
            return action_id
    return None


def _best_targeted_attack_action(
    legal_action_ids: Sequence[str],
    structured_observation: Mapping[str, Any],
) -> str | None:
    combat = structured_observation.get("combat")
    if not isinstance(combat, Mapping):
        return None

    hand = combat.get("hand")
    enemies = combat.get("enemies")
    if not isinstance(hand, Sequence) or not isinstance(enemies, Sequence):
        return None

    targetable_indices = {
        enemy_index
        for enemy_index, enemy in enumerate(enemies)
        if isinstance(enemy, Mapping) and bool(enemy.get("is_targetable", False))
    }
    if not targetable_indices:
        return None

    best_action_id: str | None = None
    best_sort_key: tuple[int, int, int] | None = None
    for action_id in legal_action_ids:
        action = action_from_id(action_id)
        if not isinstance(action, PlayCardAction) or action.target is None:
            continue
        if action.hand_index >= len(hand) or action.target.monster_index not in targetable_indices:
            continue

        raw_card = hand[action.hand_index]
        if not isinstance(raw_card, Mapping):
            continue

        enemy = enemies[action.target.monster_index]
        if not isinstance(enemy, Mapping):
            continue

        sort_key = (
            _as_int(enemy.get("current_hp"), default=0),
            _card_priority(raw_card),
            action.hand_index,
        )
        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_action_id = action_id
    return best_action_id


def _first_play_card_without_target(legal_action_ids: Sequence[str]) -> str | None:
    for action_id in legal_action_ids:
        action = action_from_id(action_id)
        if isinstance(action, PlayCardAction) and action.target is None:
            return action_id
    return None


def _incoming_damage(raw_enemies: Any) -> int:
    if not isinstance(raw_enemies, Sequence):
        return 0

    total = 0
    for enemy in raw_enemies:
        if not isinstance(enemy, Mapping):
            continue
        if not bool(enemy.get("is_targetable", False)):
            continue
        base_damage = _as_int(enemy.get("intent_base_damage"))
        hits = max(1, _as_int(enemy.get("intent_hits"), default=1))
        total += base_damage * hits
    return total


def _is_defensive_card(raw_card: Mapping[str, Any]) -> bool:
    haystack = " ".join(
        str(value).lower()
        for value in (raw_card.get("name"), raw_card.get("card_id"), raw_card.get("id"))
        if value is not None
    )
    return any(token in haystack for token in _DEFENSIVE_CARD_TOKENS)


def _card_priority(raw_card: Mapping[str, Any]) -> int:
    haystack = " ".join(
        str(value).lower()
        for value in (raw_card.get("name"), raw_card.get("card_id"), raw_card.get("id"))
        if value is not None
    )
    if "bash" in haystack:
        return 0
    if "strike" in haystack:
        return 1
    return 2


def _as_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return default
