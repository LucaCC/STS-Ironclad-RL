"""Python-side bridge scaffolding for a live Slay the Spire session."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .protocol import (
    ActionCommand,
    BridgeEnvelope,
    BridgeMessageType,
    BridgeSessionHello,
    GameStateSnapshot,
)


class BridgeTransport:
    """Abstract transport contract used by the live-game bridge."""

    def open(self, config: "BridgeConfig") -> None:
        """Open the underlying transport."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the underlying transport."""
        raise NotImplementedError

    def send(self, envelope: BridgeEnvelope) -> None:
        """Send one envelope to the communication layer."""
        raise NotImplementedError

    def receive(self) -> BridgeEnvelope | None:
        """Receive one envelope from the communication layer, if available."""
        raise NotImplementedError


@dataclass(frozen=True)
class BridgeConfig:
    """Host-specific bridge configuration supplied by the user."""

    host: str = "127.0.0.1"
    port: int = 8080
    connect_timeout_seconds: float = 5.0
    receive_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class LiveGameBridgeSession:
    """Session metadata tracked by the bridge after connect."""

    session_id: str
    config: BridgeConfig


class LiveGameBridge:
    """Manage a bridge session without assuming a concrete transport."""

    def __init__(self, transport: BridgeTransport, config: BridgeConfig | None = None) -> None:
        self._transport = transport
        self._config = config or BridgeConfig()
        self._session: LiveGameBridgeSession | None = None

    @property
    def session(self) -> LiveGameBridgeSession | None:
        """Return the active bridge session if connected."""
        return self._session

    def connect(self) -> LiveGameBridgeSession:
        """Open the transport and advertise a new bridge session."""
        if self._session is not None:
            return self._session

        self._transport.open(self._config)
        session = LiveGameBridgeSession(session_id=str(uuid4()), config=self._config)
        hello = BridgeSessionHello(session_id=session.session_id)
        self._transport.send(BridgeEnvelope.from_message(BridgeMessageType.SESSION_HELLO, hello))
        self._session = session
        return session

    def close(self) -> None:
        """Close the active bridge session."""
        if self._session is None:
            return

        self._transport.close()
        self._session = None

    def request_state(self) -> None:
        """Ask the communication layer for the latest structured game state."""
        session = self._require_session()
        self._transport.send(
            BridgeEnvelope(
                message_type=BridgeMessageType.REQUEST_STATE,
                payload={"session_id": session.session_id},
            )
        )

    def send_action(self, action: ActionCommand) -> None:
        """Send a typed action command to the communication layer."""
        session = self._require_session()
        if action.session_id != session.session_id:
            msg = "action.session_id must match the active bridge session"
            raise ValueError(msg)

        self._transport.send(BridgeEnvelope.from_message(BridgeMessageType.ACTION_COMMAND, action))

    def receive_state(self) -> GameStateSnapshot | None:
        """Receive the next game-state snapshot if one is available."""
        session = self._require_session()
        envelope = self._transport.receive()
        if envelope is None:
            return None
        if envelope.message_type is not BridgeMessageType.GAME_STATE:
            msg = (
                f"expected {BridgeMessageType.GAME_STATE.value}, got {envelope.message_type.value}"
            )
            raise ValueError(msg)
        if envelope.payload.get("session_id") != session.session_id:
            msg = "game_state session_id must match the active bridge session"
            raise ValueError(msg)

        return GameStateSnapshot(**envelope.payload)

    def _require_session(self) -> LiveGameBridgeSession:
        if self._session is None:
            msg = "bridge session is not connected"
            raise RuntimeError(msg)
        return self._session
