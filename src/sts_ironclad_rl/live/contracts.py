"""Minimal live-game RL contracts built on top of the bridge package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..integration import ActionCommand, GameStateSnapshot


@dataclass(frozen=True)
class EncodedObservation:
    """Policy-facing representation derived from one live game snapshot."""

    snapshot: GameStateSnapshot
    legal_action_ids: tuple[str, ...]
    features: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDecision:
    """Policy-selected action in the repo-level live action namespace."""

    action_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObservationEncoder(Protocol):
    """Translate raw bridge snapshots into policy-facing observations."""

    def encode(self, snapshot: GameStateSnapshot) -> EncodedObservation:
        """Encode a live snapshot for policy consumption."""


class ActionContract(Protocol):
    """Own the policy-facing action namespace and bridge mapping."""

    def legal_action_ids(self, snapshot: GameStateSnapshot) -> tuple[str, ...]:
        """Return the legal policy-facing actions for a snapshot."""

    def to_command(self, session_id: str, decision: ActionDecision) -> ActionCommand:
        """Translate one policy decision into a bridge action command."""


class Policy(Protocol):
    """Minimal policy contract for live-game decisions."""

    name: str

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        """Choose one action for the provided encoded observation."""


@dataclass(frozen=True)
class RawStateObservationEncoder:
    """Conservative default encoder that exposes raw bridge state directly."""

    include_snapshot_metadata: bool = True

    def encode(self, snapshot: GameStateSnapshot) -> EncodedObservation:
        metadata: dict[str, Any] = {}
        if self.include_snapshot_metadata:
            metadata = {
                "screen_state": snapshot.screen_state,
                "in_combat": snapshot.in_combat,
                "floor": snapshot.floor,
                "act": snapshot.act,
            }

        return EncodedObservation(
            snapshot=snapshot,
            legal_action_ids=snapshot.available_actions,
            features=dict(snapshot.raw_state),
            metadata=metadata,
        )


@dataclass(frozen=True)
class SnapshotActionContract:
    """Minimal action contract that mirrors the bridge's available actions."""

    def legal_action_ids(self, snapshot: GameStateSnapshot) -> tuple[str, ...]:
        return snapshot.available_actions

    def to_command(self, session_id: str, decision: ActionDecision) -> ActionCommand:
        return ActionCommand(
            session_id=session_id,
            command=decision.action_id,
            arguments=dict(decision.arguments),
        )

    def validate(self, snapshot: GameStateSnapshot, decision: ActionDecision) -> None:
        """Reject actions that are not exposed by the current snapshot."""
        if decision.action_id not in snapshot.available_actions:
            msg = f"illegal action for snapshot: {decision.action_id}"
            raise ValueError(msg)

    def to_validated_command(
        self,
        snapshot: GameStateSnapshot,
        decision: ActionDecision,
    ) -> ActionCommand:
        """Validate against the snapshot before producing a bridge command."""
        self.validate(snapshot, decision)
        return self.to_command(snapshot.session_id, decision)
