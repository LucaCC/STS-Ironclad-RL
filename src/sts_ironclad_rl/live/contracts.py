"""Minimal live-game RL contracts built on top of the bridge package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ..integration import ActionCommand, GameStateSnapshot


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds")


def _snapshot_from_dict(data: Mapping[str, Any]) -> GameStateSnapshot:
    return GameStateSnapshot(
        session_id=str(data["session_id"]),
        screen_state=str(data["screen_state"]),
        available_actions=tuple(data.get("available_actions", ())),
        in_combat=bool(data["in_combat"]),
        floor=data.get("floor"),
        act=data.get("act"),
        raw_state=dict(data.get("raw_state", {})),
    )


def _action_command_from_dict(data: Mapping[str, Any]) -> ActionCommand:
    return ActionCommand(
        session_id=str(data["session_id"]),
        command=str(data["command"]),
        arguments=dict(data.get("arguments", {})),
    )


@dataclass(frozen=True)
class EncodedObservation:
    """Policy-facing representation derived from one live game snapshot."""

    snapshot: GameStateSnapshot
    legal_action_ids: tuple[str, ...]
    features: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": asdict(self.snapshot),
            "legal_action_ids": list(self.legal_action_ids),
            "features": dict(self.features),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EncodedObservation":
        return cls(
            snapshot=_snapshot_from_dict(data["snapshot"]),
            legal_action_ids=tuple(data.get("legal_action_ids", ())),
            features=dict(data.get("features", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ActionDecision:
    """Policy-selected action in the repo-level live action namespace."""

    action_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "arguments": dict(self.arguments),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionDecision":
        return cls(
            action_id=str(data["action_id"]),
            arguments=dict(data.get("arguments", {})),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_action(cls, action: Any) -> "ActionDecision":
        """Build a decision from any canonical action object with an ``action_id``."""
        action_id = getattr(action, "action_id", None)
        if not isinstance(action_id, str):
            msg = "action must expose a string action_id"
            raise TypeError(msg)
        return cls(action_id=action_id)


@dataclass(frozen=True)
class ReplayFailure:
    """Structured interruption or error metadata for a replay step."""

    stage: str
    message: str
    error_type: str | None = None
    recoverable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "error_type": self.error_type,
            "recoverable": self.recoverable,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayFailure":
        return cls(
            stage=str(data["stage"]),
            message=str(data["message"]),
            error_type=data.get("error_type"),
            recoverable=bool(data.get("recoverable", False)),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class ReplayEntry:
    """Structured replay record for one live rollout step."""

    session_id: str
    step_index: int
    observation: EncodedObservation
    recorded_at: str = field(default_factory=_utc_now_iso)
    state_reference: Mapping[str, Any] = field(default_factory=dict)
    legal_action_ids: tuple[str, ...] = field(default_factory=tuple)
    action: ActionDecision | None = None
    command: ActionCommand | None = None
    reward: float | None = None
    terminal: bool = False
    outcome: str | None = None
    failure: ReplayFailure | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observation.snapshot.session_id != self.session_id:
            msg = "observation snapshot session_id must match replay entry session_id"
            raise ValueError(msg)
        if self.command is not None and self.command.session_id != self.session_id:
            msg = "command session_id must match replay entry session_id"
            raise ValueError(msg)
        legal_action_ids = self.legal_action_ids or self.observation.legal_action_ids
        if tuple(legal_action_ids) != tuple(self.observation.legal_action_ids):
            msg = "legal_action_ids must match observation.legal_action_ids"
            raise ValueError(msg)
        object.__setattr__(self, "legal_action_ids", tuple(legal_action_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "step_index": self.step_index,
            "recorded_at": self.recorded_at,
            "state_reference": dict(self.state_reference),
            "observation": self.observation.to_dict(),
            "legal_action_ids": list(self.legal_action_ids),
            "action": None if self.action is None else self.action.to_dict(),
            "command": None if self.command is None else asdict(self.command),
            "reward": self.reward,
            "terminal": self.terminal,
            "outcome": self.outcome,
            "failure": None if self.failure is None else self.failure.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayEntry":
        action = data.get("action")
        command = data.get("command")
        failure = data.get("failure")
        return cls(
            session_id=str(data["session_id"]),
            step_index=int(data["step_index"]),
            observation=EncodedObservation.from_dict(data["observation"]),
            recorded_at=str(data["recorded_at"]),
            state_reference=dict(data.get("state_reference", {})),
            legal_action_ids=tuple(data.get("legal_action_ids", ())),
            action=None if action is None else ActionDecision.from_dict(action),
            command=None if command is None else _action_command_from_dict(command),
            reward=data.get("reward"),
            terminal=bool(data.get("terminal", False)),
            outcome=data.get("outcome"),
            failure=None if failure is None else ReplayFailure.from_dict(failure),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RolloutResult:
    """Outcome of a single live rollout or bounded session slice."""

    session_id: str
    entries: tuple[ReplayEntry, ...]
    terminal: bool
    outcome: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationCase:
    """Named evaluation scenario for a live rollout batch."""

    name: str
    max_steps: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation output built on top of rollout results."""

    policy_name: str
    case_name: str
    episode_count: int
    completion_rate: float
    mean_steps: float
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


class ReplaySink(Protocol):
    """Write structured replay records for later analysis."""

    def log(self, entry: ReplayEntry) -> None:
        """Persist one replay entry."""


class Policy(Protocol):
    """Minimal policy contract for live-game rollouts."""

    name: str

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        """Choose one action for the provided encoded observation."""


class RolloutRunner(Protocol):
    """Run the bridge-facing control loop for one policy."""

    def run_episode(
        self,
        *,
        policy: Policy,
        evaluation_case: EvaluationCase | None = None,
    ) -> RolloutResult:
        """Run one live episode or bounded session slice."""


@dataclass(frozen=True)
class RawStateObservationEncoder:
    """Conservative default encoder that exposes raw bridge state directly."""

    include_snapshot_metadata: bool = True

    def encode(self, snapshot: GameStateSnapshot) -> EncodedObservation:
        from .actions import CommunicationModActionContract

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
            legal_action_ids=CommunicationModActionContract().legal_action_ids(snapshot),
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
