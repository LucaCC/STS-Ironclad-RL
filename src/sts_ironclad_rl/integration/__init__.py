"""Live-game integration scaffolding for Slay the Spire mod communication."""

from .bridge import BridgeConfig, BridgeTransport, LiveGameBridge, LiveGameBridgeSession
from .communication_mod import (
    CommunicationModBridgeHelper,
    SocketBridgeTransport,
    build_transport,
    compute_snapshot_fingerprint,
    helper_main,
    translate_action_command_to_comm,
    translate_comm_message_to_snapshot,
)
from .logger import JsonlTrajectoryLogger
from .protocol import (
    ActionCommand,
    BridgeEnvelope,
    BridgeMessageType,
    BridgeSessionHello,
    GameStateSnapshot,
    TrajectoryEntry,
)

__all__ = [
    "ActionCommand",
    "BridgeConfig",
    "BridgeEnvelope",
    "BridgeMessageType",
    "BridgeSessionHello",
    "CommunicationModBridgeHelper",
    "BridgeTransport",
    "GameStateSnapshot",
    "JsonlTrajectoryLogger",
    "LiveGameBridge",
    "LiveGameBridgeSession",
    "SocketBridgeTransport",
    "TrajectoryEntry",
    "build_transport",
    "compute_snapshot_fingerprint",
    "helper_main",
    "translate_action_command_to_comm",
    "translate_comm_message_to_snapshot",
]
