"""Canonical live-game actions and CommunicationMod command mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from ..integration import ActionCommand, GameStateSnapshot
from .contracts import ActionDecision


class LiveActionType(StrEnum):
    """Stable policy-facing action identifiers."""

    PLAY_CARD = "play_card"
    END_TURN = "end_turn"
    CHOOSE = "choose"
    PROCEED = "proceed"
    LEAVE = "leave"


@dataclass(frozen=True)
class MonsterTarget:
    """Zero-based monster target for a combat action."""

    monster_index: int

    def __post_init__(self) -> None:
        if self.monster_index < 0:
            msg = "monster_index must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class PlayCardAction:
    """Play a card from the current hand."""

    hand_index: int
    target: MonsterTarget | None = None

    def __post_init__(self) -> None:
        if self.hand_index < 0:
            msg = "hand_index must be non-negative"
            raise ValueError(msg)

    @property
    def action_type(self) -> LiveActionType:
        return LiveActionType.PLAY_CARD

    @property
    def action_id(self) -> str:
        if self.target is None:
            return f"{self.action_type.value}:{self.hand_index}"
        return f"{self.action_type.value}:{self.hand_index}:{self.target.monster_index}"


@dataclass(frozen=True)
class EndTurnAction:
    """End the current combat turn."""

    @property
    def action_type(self) -> LiveActionType:
        return LiveActionType.END_TURN

    @property
    def action_id(self) -> str:
        return self.action_type.value


@dataclass(frozen=True)
class ChooseAction:
    """Choose a zero-based option from the current choice list."""

    choice_index: int

    def __post_init__(self) -> None:
        if self.choice_index < 0:
            msg = "choice_index must be non-negative"
            raise ValueError(msg)

    @property
    def action_type(self) -> LiveActionType:
        return LiveActionType.CHOOSE

    @property
    def action_id(self) -> str:
        return f"{self.action_type.value}:{self.choice_index}"


@dataclass(frozen=True)
class ProceedAction:
    """Confirm and advance the current screen."""

    @property
    def action_type(self) -> LiveActionType:
        return LiveActionType.PROCEED

    @property
    def action_id(self) -> str:
        return self.action_type.value


@dataclass(frozen=True)
class LeaveAction:
    """Leave the current screen when the game exposes that option."""

    @property
    def action_type(self) -> LiveActionType:
        return LiveActionType.LEAVE

    @property
    def action_id(self) -> str:
        return self.action_type.value


LiveAction: TypeAlias = PlayCardAction | EndTurnAction | ChooseAction | ProceedAction | LeaveAction


def action_to_id(action: LiveAction) -> str:
    """Return the stable identifier for a canonical action object."""
    return action.action_id


def action_from_id(action_id: str) -> LiveAction:
    """Parse one stable action identifier into a canonical action object."""
    parts = action_id.split(":")
    action_name = parts[0]

    if action_name == LiveActionType.END_TURN.value and len(parts) == 1:
        return EndTurnAction()
    if action_name == LiveActionType.PROCEED.value and len(parts) == 1:
        return ProceedAction()
    if action_name == LiveActionType.LEAVE.value and len(parts) == 1:
        return LeaveAction()
    if action_name == LiveActionType.CHOOSE.value and len(parts) == 2:
        return ChooseAction(choice_index=_parse_index(parts[1], label="choice_index"))
    if action_name == LiveActionType.PLAY_CARD.value and len(parts) == 2:
        return PlayCardAction(hand_index=_parse_index(parts[1], label="hand_index"))
    if action_name == LiveActionType.PLAY_CARD.value and len(parts) == 3:
        return PlayCardAction(
            hand_index=_parse_index(parts[1], label="hand_index"),
            target=MonsterTarget(_parse_index(parts[2], label="monster_index")),
        )

    msg = f"unsupported action_id: {action_id}"
    raise ValueError(msg)


@dataclass(frozen=True)
class CommunicationModActionContract:
    """Validate live actions against a snapshot and map them into bridge commands."""

    def legal_action_ids(self, snapshot: GameStateSnapshot) -> tuple[str, ...]:
        return tuple(action.action_id for action in self.legal_actions(snapshot))

    def legal_actions(self, snapshot: GameStateSnapshot) -> tuple[LiveAction, ...]:
        actions: list[LiveAction] = []
        available = set(snapshot.available_actions)

        if "choose" in available:
            for choice_index in range(len(_choice_list(snapshot))):
                actions.append(ChooseAction(choice_index=choice_index))

        if "proceed" in available:
            actions.append(ProceedAction())

        if "leave" in available:
            actions.append(LeaveAction())

        if "play" in available:
            combat_state = _combat_state(snapshot)
            for hand_index, card in enumerate(_hand(combat_state)):
                if not bool(card.get("is_playable", False)):
                    continue
                if bool(card.get("has_target", False)):
                    for monster_index in _living_monster_indices(combat_state):
                        actions.append(
                            PlayCardAction(
                                hand_index=hand_index,
                                target=MonsterTarget(monster_index),
                            )
                        )
                else:
                    actions.append(PlayCardAction(hand_index=hand_index))

        if "end" in available:
            actions.append(EndTurnAction())

        return tuple(actions)

    def validate_action(self, snapshot: GameStateSnapshot, action: LiveAction) -> None:
        available = set(snapshot.available_actions)

        if isinstance(action, PlayCardAction):
            self._validate_play_card(snapshot, action, available)
            return
        if isinstance(action, EndTurnAction):
            self._require_available(available, "end", action.action_id)
            if not snapshot.in_combat:
                msg = "end_turn is only valid while in combat"
                raise ValueError(msg)
            return
        if isinstance(action, ChooseAction):
            self._require_available(available, "choose", action.action_id)
            choice_list = _choice_list(snapshot)
            if action.choice_index >= len(choice_list):
                msg = f"choice_index out of range: {action.choice_index}"
                raise ValueError(msg)
            return
        if isinstance(action, ProceedAction):
            self._require_available(available, "proceed", action.action_id)
            return
        if isinstance(action, LeaveAction):
            self._require_available(available, "leave", action.action_id)
            return

        msg = f"unsupported action type: {type(action).__name__}"
        raise TypeError(msg)

    def to_action(self, action_id: str) -> LiveAction:
        return action_from_id(action_id)

    def to_command(self, session_id: str, decision: ActionDecision) -> ActionCommand:
        if decision.arguments:
            msg = "canonical live actions do not accept free-form decision.arguments"
            raise ValueError(msg)
        action = self.to_action(decision.action_id)
        return self.action_to_command(session_id=session_id, action=action)

    def action_to_command(self, session_id: str, action: LiveAction) -> ActionCommand:
        if isinstance(action, PlayCardAction):
            arguments = {"card_index": action.hand_index + 1}
            if action.target is not None:
                arguments["target_index"] = action.target.monster_index
            return ActionCommand(session_id=session_id, command="play", arguments=arguments)
        if isinstance(action, EndTurnAction):
            return ActionCommand(session_id=session_id, command="end")
        if isinstance(action, ChooseAction):
            return ActionCommand(
                session_id=session_id,
                command="choose",
                arguments={"choice_index": action.choice_index},
            )
        if isinstance(action, ProceedAction):
            return ActionCommand(session_id=session_id, command="proceed")
        if isinstance(action, LeaveAction):
            return ActionCommand(session_id=session_id, command="leave")

        msg = f"unsupported action type: {type(action).__name__}"
        raise TypeError(msg)

    def to_validated_command(
        self,
        snapshot: GameStateSnapshot,
        decision: ActionDecision,
    ) -> ActionCommand:
        if decision.arguments:
            msg = "canonical live actions do not accept free-form decision.arguments"
            raise ValueError(msg)
        action = self.to_action(decision.action_id)
        self.validate_action(snapshot, action)
        return self.action_to_command(session_id=snapshot.session_id, action=action)

    def _validate_play_card(
        self,
        snapshot: GameStateSnapshot,
        action: PlayCardAction,
        available: set[str],
    ) -> None:
        self._require_available(available, "play", action.action_id)
        if not snapshot.in_combat:
            msg = "play_card is only valid while in combat"
            raise ValueError(msg)

        hand = _hand(_combat_state(snapshot))
        if action.hand_index >= len(hand):
            msg = f"hand_index out of range: {action.hand_index}"
            raise ValueError(msg)

        card = hand[action.hand_index]
        if not bool(card.get("is_playable", False)):
            msg = f"card is not playable at hand_index={action.hand_index}"
            raise ValueError(msg)

        card_requires_target = bool(card.get("has_target", False))
        if card_requires_target and action.target is None:
            msg = "targeted card requires a target"
            raise ValueError(msg)
        if not card_requires_target and action.target is not None:
            msg = "untargeted card does not accept a target"
            raise ValueError(msg)
        if action.target is not None and action.target.monster_index not in _living_monster_indices(
            _combat_state(snapshot)
        ):
            msg = f"invalid target_index: {action.target.monster_index}"
            raise ValueError(msg)

    def _require_available(self, available: set[str], command_name: str, action_id: str) -> None:
        if command_name not in available:
            msg = f"action is unavailable for snapshot: {action_id}"
            raise ValueError(msg)


def _parse_index(value: str, *, label: str) -> int:
    try:
        index = int(value)
    except ValueError as exc:
        msg = f"{label} must be an integer"
        raise ValueError(msg) from exc
    if index < 0:
        msg = f"{label} must be non-negative"
        raise ValueError(msg)
    return index


def _combat_state(snapshot: GameStateSnapshot) -> dict[str, object]:
    combat_state = snapshot.raw_state.get("combat_state")
    if isinstance(combat_state, dict):
        return combat_state
    return {}


def _hand(combat_state: dict[str, object]) -> list[dict[str, object]]:
    hand = combat_state.get("hand")
    if not isinstance(hand, list):
        return []
    return [card for card in hand if isinstance(card, dict)]


def _living_monster_indices(combat_state: dict[str, object]) -> tuple[int, ...]:
    monsters = combat_state.get("monsters")
    if not isinstance(monsters, list):
        return ()

    living: list[int] = []
    for index, monster in enumerate(monsters):
        if not isinstance(monster, dict):
            continue
        if bool(monster.get("is_gone", False)):
            continue
        if bool(monster.get("half_dead", False)):
            continue
        current_hp = monster.get("current_hp")
        if isinstance(current_hp, int) and current_hp <= 0:
            continue
        living.append(index)
    return tuple(living)


def _choice_list(snapshot: GameStateSnapshot) -> tuple[object, ...]:
    choices = snapshot.raw_state.get("choice_list")
    if not isinstance(choices, list):
        return ()
    return tuple(choice for choice in choices if isinstance(choice, str))
