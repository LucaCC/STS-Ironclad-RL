"""Live-game integration scaffolding for Slay the Spire mod communication."""

from .bridge import BridgeConfig, BridgeTransport, LiveGameBridge, LiveGameBridgeSession
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
    "BridgeTransport",
    "GameStateSnapshot",
    "JsonlTrajectoryLogger",
    "LiveGameBridge",
    "LiveGameBridgeSession",
    "TrajectoryEntry",
]
