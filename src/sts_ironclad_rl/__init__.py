"""Top-level package for the Slay the Spire RL research stack."""

from .bootstrap import ProjectInfo, get_project_info
from .env import (
    Action,
    CombatantState,
    CombatEnvironment,
    CombatState,
    DrawResult,
    EncounterConfig,
    InvalidActionError,
    PileState,
    StepResult,
    create_initial_combat_state,
    draw_cards,
)
from .integration import (
    ActionCommand,
    BridgeConfig,
    BridgeEnvelope,
    BridgeMessageType,
    BridgeSessionHello,
    BridgeTransport,
    GameStateSnapshot,
    JsonlTrajectoryLogger,
    LiveGameBridge,
    LiveGameBridgeSession,
    TrajectoryEntry,
)

__all__ = [
    "Action",
    "ActionCommand",
    "BridgeConfig",
    "BridgeEnvelope",
    "BridgeMessageType",
    "BridgeSessionHello",
    "BridgeTransport",
    "CombatState",
    "CombatantState",
    "CombatEnvironment",
    "DrawResult",
    "EncounterConfig",
    "GameStateSnapshot",
    "InvalidActionError",
    "JsonlTrajectoryLogger",
    "LiveGameBridge",
    "LiveGameBridgeSession",
    "PileState",
    "ProjectInfo",
    "StepResult",
    "TrajectoryEntry",
    "create_initial_combat_state",
    "draw_cards",
    "get_project_info",
]
__version__ = "0.1.0"
