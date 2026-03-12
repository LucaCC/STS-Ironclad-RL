"""Typed protocol contracts for the live-game bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class BridgeMessageType(StrEnum):
    """High-level message types exchanged across the live-game bridge."""

    SESSION_HELLO = "session_hello"
    REQUEST_STATE = "request_state"
    GAME_STATE = "game_state"
    ACTION_COMMAND = "action_command"
    ACK = "ack"
    ERROR = "error"
    TRAJECTORY_ENTRY = "trajectory_entry"


@dataclass(frozen=True)
class BridgeSessionHello:
    """Session metadata advertised by the Python-side bridge."""

    session_id: str
    protocol_version: str = "0.1"
    client_name: str = "sts_ironclad_rl"


@dataclass(frozen=True)
class GameStateSnapshot:
    """Structured state observed from the live game or mod layer."""

    session_id: str
    screen_state: str
    available_actions: tuple[str, ...]
    in_combat: bool
    floor: int | None = None
    act: int | None = None
    raw_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionCommand:
    """Action request sent from Python to the communication layer."""

    session_id: str
    command: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryEntry:
    """Observation or action trace captured for later replay and debugging."""

    session_id: str
    step_index: int
    observation: GameStateSnapshot
    action: ActionCommand | None = None
    reward: float | None = None
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgeEnvelope:
    """Transport-neutral wrapper around a typed bridge payload."""

    message_type: BridgeMessageType
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for transport or logging."""
        return {"message_type": self.message_type.value, "payload": self.payload}

    @classmethod
    def from_message(
        cls,
        message_type: BridgeMessageType,
        message: BridgeSessionHello | GameStateSnapshot | ActionCommand | TrajectoryEntry,
    ) -> "BridgeEnvelope":
        """Wrap a typed protocol message in a transport-neutral envelope."""
        return cls(message_type=message_type, payload=asdict(message))
