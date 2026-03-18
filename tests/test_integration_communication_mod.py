from __future__ import annotations

import json
import socket
import threading

import pytest

from sts_ironclad_rl.integration import (
    ActionCommand,
    BridgeConfig,
    BridgeEnvelope,
    BridgeMessageType,
    BridgeSessionHello,
    CommunicationModBridgeHelper,
    SocketBridgeTransport,
    translate_action_command_to_comm,
    translate_comm_message_to_snapshot,
)


def test_translate_comm_message_to_snapshot_flattens_game_state() -> None:
    message = {
        "available_commands": ["play", "end", "state"],
        "in_game": True,
        "ready_for_command": True,
        "game_state": {
            "screen_type": "COMBAT",
            "floor": 7,
            "act": 2,
            "room_phase": "COMBAT",
            "choice_list": ["take", "skip"],
            "combat_state": {
                "turn": 2,
                "player": {"current_hp": 51, "max_hp": 80, "block": 4, "energy": 2},
            },
        },
    }

    snapshot = translate_comm_message_to_snapshot(message, session_id="session-1")

    assert snapshot is not None
    assert snapshot.session_id == "session-1"
    assert snapshot.screen_state == "COMBAT"
    assert snapshot.available_actions == ("play", "end", "state")
    assert snapshot.in_combat is True
    assert snapshot.floor == 7
    assert snapshot.act == 2
    assert snapshot.raw_state["combat_state"]["turn"] == 2
    assert snapshot.raw_state["choice_list"] == ["take", "skip"]
    assert snapshot.raw_state["ready_for_command"] is True


def test_translate_action_command_to_comm_formats_supported_commands() -> None:
    assert (
        translate_action_command_to_comm(
            ActionCommand(session_id="session-1", command="play", arguments={"card_index": 2})
        )
        == "PLAY 2"
    )
    assert (
        translate_action_command_to_comm(
            ActionCommand(
                session_id="session-1",
                command="play",
                arguments={"card_index": 1, "target_index": 0},
            )
        )
        == "PLAY 1 0"
    )
    assert (
        translate_action_command_to_comm(
            ActionCommand(session_id="session-1", command="choose", arguments={"choice_index": 3})
        )
        == "CHOOSE 3"
    )
    assert (
        translate_action_command_to_comm(ActionCommand(session_id="session-1", command="end"))
        == "END"
    )
    assert (
        translate_action_command_to_comm(ActionCommand(session_id="session-1", command="proceed"))
        == "PROCEED"
    )
    assert (
        translate_action_command_to_comm(ActionCommand(session_id="session-1", command="leave"))
        == "LEAVE"
    )


def test_helper_waits_for_post_action_snapshot() -> None:
    helper = CommunicationModBridgeHelper(timeout_seconds=0.2)
    helper.handle_envelope(
        BridgeEnvelope.from_message(
            BridgeMessageType.SESSION_HELLO,
            BridgeSessionHello(session_id="session-1"),
        )
    )
    helper.ingest_mod_message(
        {
            "available_commands": ["play", "state"],
            "in_game": True,
            "ready_for_command": True,
            "game_state": {
                "screen_type": "COMBAT",
                "room_phase": "COMBAT",
                "combat_state": {"turn": 1},
            },
        }
    )

    def submit_action() -> None:
        helper.handle_envelope(
            BridgeEnvelope.from_message(
                BridgeMessageType.ACTION_COMMAND,
                ActionCommand(
                    session_id="session-1",
                    command="play",
                    arguments={"card_index": 1},
                ),
            )
        )

    thread = threading.Thread(target=submit_action)
    thread.start()

    assert (
        helper.ingest_mod_message(
            {
                "available_commands": ["play", "state"],
                "in_game": True,
                "ready_for_command": True,
                "game_state": {
                    "screen_type": "COMBAT",
                    "room_phase": "COMBAT",
                    "combat_state": {"turn": 1},
                },
            },
        )
        == "PLAY 1"
    )
    assert (
        helper.handle_envelope(
            BridgeEnvelope(
                message_type=BridgeMessageType.REQUEST_STATE,
                payload={"session_id": "session-1"},
            )
        )
        is None
    )

    helper.ingest_mod_message(
        {
            "available_commands": ["end", "state"],
            "in_game": True,
            "ready_for_command": True,
            "game_state": {
                "screen_type": "COMBAT",
                "room_phase": "COMBAT",
                "combat_state": {"turn": 2},
            },
        }
    )
    response = helper.handle_envelope(
        BridgeEnvelope(
            message_type=BridgeMessageType.REQUEST_STATE,
            payload={"session_id": "session-1"},
        )
    )
    thread.join(timeout=1.0)

    assert response is not None
    assert response.message_type is BridgeMessageType.GAME_STATE
    assert response.payload["raw_state"]["combat_state"]["turn"] == 2


def test_socket_bridge_transport_sends_and_receives_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted: dict[str, object] = {}
    client_socket, server_socket = socket.socketpair()

    def fake_create_connection(address, timeout):  # noqa: ANN001,ANN201
        accepted["address"] = address
        accepted["timeout"] = timeout
        return client_socket

    def run_server() -> None:
        with server_socket:
            reader = server_socket.makefile("r", encoding="utf-8")
            accepted["line"] = reader.readline()
            response = BridgeEnvelope(
                message_type=BridgeMessageType.GAME_STATE,
                payload={
                    "session_id": "session-1",
                    "screen_state": "COMBAT",
                    "available_actions": ["end"],
                    "in_combat": True,
                    "floor": 1,
                    "act": 1,
                    "raw_state": {"combat_state": {"turn": 1}},
                },
            )
            server_socket.sendall(
                (json.dumps(response.to_dict(), sort_keys=True) + "\n").encode("utf-8")
            )
            reader.close()

    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    transport = SocketBridgeTransport()
    transport.open(BridgeConfig(host="127.0.0.1", port=8080, connect_timeout_seconds=1.0))
    transport.send(
        BridgeEnvelope.from_message(
            BridgeMessageType.SESSION_HELLO,
            BridgeSessionHello(session_id="session-1"),
        )
    )
    envelope = transport.receive()
    transport.close()
    server_thread.join(timeout=1.0)

    assert accepted["line"] is not None
    assert accepted["address"] == ("127.0.0.1", 8080)
    sent = json.loads(str(accepted["line"]))
    assert sent["message_type"] == "session_hello"
    assert envelope is not None
    assert envelope.message_type is BridgeMessageType.GAME_STATE
    assert envelope.payload["available_actions"] == ("end",)
