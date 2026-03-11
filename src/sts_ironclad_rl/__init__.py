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

__all__ = [
    "Action",
    "CombatState",
    "CombatantState",
    "CombatEnvironment",
    "DrawResult",
    "EncounterConfig",
    "InvalidActionError",
    "PileState",
    "ProjectInfo",
    "StepResult",
    "create_initial_combat_state",
    "draw_cards",
    "get_project_info",
]
__version__ = "0.1.0"
