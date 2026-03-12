from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from sts_ironclad_rl.integration import (
    ActionCommand,
    BridgeConfig,
    BridgeEnvelope,
    BridgeMessageType,
    GameStateSnapshot,
    LiveGameBridge,
)
from sts_ironclad_rl.integration.bridge import BridgeTransport


@dataclass
class FakeTransport(BridgeTransport):
    opened_with: BridgeConfig | None = None
    closed: bool = False
    sent: list[BridgeEnvelope] = field(default_factory=list)
    received: list[BridgeEnvelope] = field(default_factory=list)

    def open(self, config: BridgeConfig) -> None:
        self.opened_with = config

    def close(self) -> None:
        self.closed = True

    def send(self, envelope: BridgeEnvelope) -> None:
        self.sent.append(envelope)

    def receive(self) -> BridgeEnvelope | None:
        if not self.received:
            return None
        return self.received.pop(0)


def test_bridge_connect_request_and_close() -> None:
    transport = FakeTransport()
    bridge = LiveGameBridge(transport=transport)

    session = bridge.connect()
    bridge.request_state()
    bridge.close()

    assert transport.opened_with == BridgeConfig()
    assert transport.sent[0].message_type is BridgeMessageType.SESSION_HELLO
    assert transport.sent[0].payload["session_id"] == session.session_id
    assert transport.sent[1] == BridgeEnvelope(
        message_type=BridgeMessageType.REQUEST_STATE,
        payload={"session_id": session.session_id},
    )
    assert transport.closed is True
    assert bridge.session is None


def test_bridge_send_action_requires_matching_session() -> None:
    transport = FakeTransport()
    bridge = LiveGameBridge(transport=transport)
    session = bridge.connect()

    bridge.send_action(ActionCommand(session_id=session.session_id, command="end_turn"))

    assert transport.sent[-1].message_type is BridgeMessageType.ACTION_COMMAND

    with pytest.raises(ValueError, match="action.session_id"):
        bridge.send_action(ActionCommand(session_id="other-session", command="end_turn"))


def test_bridge_receive_state_rejects_unexpected_messages() -> None:
    transport = FakeTransport(
        received=[
            BridgeEnvelope(
                message_type=BridgeMessageType.GAME_STATE,
                payload={
                    "session_id": "session-1",
                    "screen_state": "COMBAT",
                    "available_actions": ("end_turn",),
                    "in_combat": True,
                    "floor": 1,
                    "act": 1,
                    "raw_state": {"player_hp": 80},
                },
            ),
            BridgeEnvelope(message_type=BridgeMessageType.ACK, payload={"ok": True}),
        ]
    )
    bridge = LiveGameBridge(transport=transport)
    bridge.connect()

    snapshot = bridge.receive_state()

    assert snapshot == GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("end_turn",),
        in_combat=True,
        floor=1,
        act=1,
        raw_state={"player_hp": 80},
    )

    with pytest.raises(ValueError, match="expected game_state"):
        bridge.receive_state()
