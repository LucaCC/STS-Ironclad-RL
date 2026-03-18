"""Concrete CommunicationMod bridge helper and socket transport."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from .bridge import BridgeConfig, BridgeTransport
from .protocol import ActionCommand, BridgeEnvelope, BridgeMessageType, GameStateSnapshot

DEFAULT_HELPER_HOST = "127.0.0.1"
DEFAULT_HELPER_PORT = 8080
DEFAULT_TIMEOUT_SECONDS = 5.0
STATE_POLL_COMMAND = "STATE"


def build_transport() -> BridgeTransport:
    """Return the default socket-backed transport for CommunicationMod."""
    return SocketBridgeTransport()


def translate_comm_message_to_snapshot(
    message: Mapping[str, Any],
    *,
    session_id: str,
) -> GameStateSnapshot | None:
    """Translate one CommunicationMod payload into the repo snapshot contract."""
    game_state = message.get("game_state")
    if not isinstance(game_state, Mapping):
        return None

    available_commands = message.get("available_commands")
    if isinstance(available_commands, list):
        available_actions = tuple(
            command for command in available_commands if isinstance(command, str)
        )
    else:
        available_actions = ()

    floor = _coerce_optional_int(game_state.get("floor"))
    act = _coerce_optional_int(game_state.get("act"))
    raw_state = dict(game_state)
    raw_state["available_commands"] = list(available_actions)
    raw_state["in_game"] = bool(message.get("in_game", False))
    raw_state["ready_for_command"] = bool(message.get("ready_for_command", False))

    error = message.get("error")
    if isinstance(error, str):
        raw_state["bridge_error"] = error

    return GameStateSnapshot(
        session_id=session_id,
        screen_state=_screen_state(game_state),
        available_actions=available_actions,
        in_combat=_is_in_combat(message=message, game_state=game_state),
        floor=floor,
        act=act,
        raw_state=raw_state,
    )


def translate_action_command_to_comm(action: ActionCommand) -> str:
    """Translate one repo action command into a CommunicationMod stdout command."""
    command_name = action.command.lower()
    arguments = action.arguments

    if command_name == "play":
        card_index = _required_int(arguments, "card_index")
        target_index = _optional_int(arguments, "target_index")
        return f"PLAY {card_index}" if target_index is None else f"PLAY {card_index} {target_index}"
    if command_name == "end":
        return "END"
    if command_name == "choose":
        choice_index = _required_int(arguments, "choice_index")
        return f"CHOOSE {choice_index}"
    if command_name == "proceed":
        return "PROCEED"
    if command_name == "leave":
        return "LEAVE"

    msg = f"unsupported CommunicationMod action command: {action.command}"
    raise ValueError(msg)


class SocketBridgeTransport(BridgeTransport):
    """Line-delimited JSON socket transport used by the live bridge scripts."""

    def __init__(self) -> None:
        self._socket: socket.socket | None = None
        self._reader: TextIO | None = None
        self._send_lock = threading.Lock()

    def open(self, config: BridgeConfig) -> None:
        if self._socket is not None:
            return
        connection = socket.create_connection(
            (config.host, config.port),
            timeout=config.connect_timeout_seconds,
        )
        connection.settimeout(config.connect_timeout_seconds)
        self._socket = connection
        self._reader = connection.makefile("r", encoding="utf-8")

    def close(self) -> None:
        reader = self._reader
        self._reader = None
        if reader is not None:
            reader.close()

        connection = self._socket
        self._socket = None
        if connection is not None:
            connection.close()

    def send(self, envelope: BridgeEnvelope) -> None:
        connection = self._require_socket()
        payload = json.dumps(envelope.to_dict(), sort_keys=True) + "\n"
        with self._send_lock:
            connection.sendall(payload.encode("utf-8"))

    def receive(self) -> BridgeEnvelope | None:
        reader = self._reader
        if reader is None:
            msg = "transport is not open"
            raise RuntimeError(msg)

        try:
            line = reader.readline()
        except TimeoutError:
            return None
        except OSError as exc:
            if isinstance(exc, socket.timeout):
                return None
            raise
        if not line:
            return None
        return _envelope_from_dict(json.loads(line))

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            msg = "transport is not open"
            raise RuntimeError(msg)
        return self._socket


@dataclass
class _PendingAction:
    command_text: str
    dispatched: bool = False
    dispatch_message_version: int | None = None


class CommunicationModBridgeHelper:
    """Bridge stdin/stdout CommunicationMod traffic to the repo bridge protocol."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HELPER_HOST,
        port: int = DEFAULT_HELPER_PORT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        stdout: TextIO | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self._stdout = sys.stdout if stdout is None else stdout
        self._state_lock = threading.Lock()
        self._state_changed = threading.Condition(self._state_lock)
        self._stdout_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._session_id: str | None = None
        self._latest_snapshot: GameStateSnapshot | None = None
        self._latest_snapshot_version = 0
        self._message_version = 0
        self._required_snapshot_version = 0
        self._pending_action: _PendingAction | None = None

    def handle_envelope(self, envelope: BridgeEnvelope) -> BridgeEnvelope | None:
        """Handle one inbound repo-side bridge envelope."""
        if envelope.message_type is BridgeMessageType.SESSION_HELLO:
            session_id = envelope.payload.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                msg = "session_hello payload must include session_id"
                raise ValueError(msg)
            with self._state_changed:
                self._session_id = session_id
                self._state_changed.notify_all()
            return None

        if envelope.message_type is BridgeMessageType.ACTION_COMMAND:
            session_id = self._require_session_id()
            action = ActionCommand(**envelope.payload)
            if action.session_id != session_id:
                msg = "action_command session_id must match the active helper session"
                raise ValueError(msg)
            self.submit_action(action)
            return None

        if envelope.message_type is BridgeMessageType.REQUEST_STATE:
            snapshot = self.wait_for_snapshot()
            if snapshot is None:
                return None
            return BridgeEnvelope.from_message(BridgeMessageType.GAME_STATE, snapshot)

        return None

    def ingest_mod_message(self, message: Mapping[str, Any]) -> str:
        """Record one CommunicationMod payload and return the next stdout command."""
        with self._state_changed:
            self._message_version += 1
            session_id = self._session_id
            if session_id is not None:
                snapshot = translate_comm_message_to_snapshot(message, session_id=session_id)
                if snapshot is not None:
                    self._latest_snapshot = snapshot
                    self._latest_snapshot_version = self._message_version

            command = STATE_POLL_COMMAND
            pending_action = self._pending_action
            if pending_action is not None and bool(message.get("ready_for_command", False)):
                command = pending_action.command_text
                pending_action.dispatched = True
                pending_action.dispatch_message_version = self._message_version
                self._required_snapshot_version = self._message_version + 1
                self._pending_action = None

            self._state_changed.notify_all()
            return command

    def submit_action(self, action: ActionCommand) -> None:
        """Queue one repo-side action and wait until the mod loop dispatches it."""
        pending_action = _PendingAction(command_text=translate_action_command_to_comm(action))
        deadline = time.monotonic() + self.timeout_seconds
        with self._state_changed:
            if self._pending_action is not None:
                msg = "helper already has a pending action"
                raise RuntimeError(msg)
            self._pending_action = pending_action
            self._state_changed.notify_all()

            while not pending_action.dispatched and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._pending_action = None
                    msg = "timed out waiting for CommunicationMod to accept the queued action"
                    raise TimeoutError(msg)
                self._state_changed.wait(timeout=remaining)

    def wait_for_snapshot(self) -> GameStateSnapshot | None:
        """Return the latest translated snapshot once it satisfies action ordering."""
        deadline = time.monotonic() + self.timeout_seconds
        with self._state_changed:
            while not self._stop_event.is_set():
                snapshot = self._latest_snapshot
                if (
                    snapshot is not None
                    and self._latest_snapshot_version >= self._required_snapshot_version
                ):
                    return snapshot

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._state_changed.wait(timeout=remaining)
        return None

    def run(self, stdin: TextIO | None = None) -> int:
        """Run the helper socket server and CommunicationMod stdin loop."""
        input_stream = sys.stdin if stdin is None else stdin
        stdin_thread = threading.Thread(
            target=self._stdin_loop,
            args=(input_stream,),
            name="communication-mod-stdin",
            daemon=True,
        )
        stdin_thread.start()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            server.settimeout(0.2)
            self._write_stdout("ready")

            while not self._stop_event.is_set():
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                except OSError as exc:
                    if self._stop_event.is_set():
                        break
                    raise exc
                client_thread = threading.Thread(
                    target=self._client_loop,
                    args=(connection,),
                    name="communication-mod-client",
                    daemon=True,
                )
                client_thread.start()

        stdin_thread.join(timeout=0.5)
        return 0

    def stop(self) -> None:
        """Signal the helper to stop serving new work."""
        self._stop_event.set()
        with self._state_changed:
            self._state_changed.notify_all()

    def _client_loop(self, connection: socket.socket) -> None:
        connection.settimeout(self.timeout_seconds)
        with connection:
            reader = connection.makefile("r", encoding="utf-8")
            try:
                for line in reader:
                    if self._stop_event.is_set():
                        return
                    payload = json.loads(line)
                    response = self.handle_envelope(_envelope_from_dict(payload))
                    if response is None:
                        continue
                    encoded = json.dumps(response.to_dict(), sort_keys=True) + "\n"
                    connection.sendall(encoded.encode("utf-8"))
            finally:
                reader.close()

    def _stdin_loop(self, stdin: TextIO) -> None:
        for raw_line in stdin:
            if self._stop_event.is_set():
                break
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                payload = {"decode_error": str(exc)}
            if not isinstance(payload, Mapping):
                payload = {"unexpected_payload": payload}

            command = self.ingest_mod_message(payload)
            self._write_stdout(command)

        self.stop()

    def _require_session_id(self) -> str:
        with self._state_lock:
            if self._session_id is None:
                msg = "session_hello must be received before other bridge messages"
                raise RuntimeError(msg)
            return self._session_id

    def _write_stdout(self, command: str) -> None:
        with self._stdout_lock:
            print(command, file=self._stdout, flush=True)


def build_helper_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the CommunicationMod helper process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HELPER_HOST, help="Socket host to bind")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_HELPER_PORT,
        help="Socket port to bind",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Socket and request timeout budget",
    )
    return parser


def helper_main(argv: list[str] | None = None) -> int:
    """Run the launchable CommunicationMod helper process."""
    args = build_helper_parser().parse_args(argv)
    helper = CommunicationModBridgeHelper(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
    )
    return helper.run()


def _envelope_from_dict(payload: Mapping[str, Any]) -> BridgeEnvelope:
    message_type = BridgeMessageType(payload["message_type"])
    envelope_payload = payload.get("payload", {})
    if not isinstance(envelope_payload, dict):
        msg = "bridge envelope payload must be an object"
        raise ValueError(msg)
    if message_type is BridgeMessageType.GAME_STATE:
        available_actions = envelope_payload.get("available_actions")
        if isinstance(available_actions, list):
            envelope_payload = dict(envelope_payload)
            envelope_payload["available_actions"] = tuple(available_actions)
    return BridgeEnvelope(message_type=message_type, payload=dict(envelope_payload))


def _screen_state(game_state: Mapping[str, Any]) -> str:
    for key in ("screen_type", "screen_name"):
        value = game_state.get(key)
        if isinstance(value, str) and value:
            return value
    return "UNKNOWN"


def _is_in_combat(*, message: Mapping[str, Any], game_state: Mapping[str, Any]) -> bool:
    if not bool(message.get("in_game", False)):
        return False
    if game_state.get("room_phase") == "COMBAT":
        return True
    return isinstance(game_state.get("combat_state"), Mapping)


def _required_int(arguments: Mapping[str, Any], key: str) -> int:
    value = _optional_int(arguments, key)
    if value is None:
        msg = f"{key} must be provided"
        raise ValueError(msg)
    return value


def _optional_int(arguments: Mapping[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None
