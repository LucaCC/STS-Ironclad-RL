"""Environment state primitives for the Slay the Spire RL stack."""

from .combat import Action, CombatEnvironment, InvalidActionError, StepResult, enemy_intent_damage
from .encoding import (
    ACTION_ORDER,
    OBSERVATION_FIELDS,
    action_to_index,
    decode_action_index,
    encode_action_mask,
    encode_observation,
    legal_action_mask,
)
from .state import (
    CombatantState,
    CombatState,
    DrawResult,
    EncounterConfig,
    PileState,
    create_initial_combat_state,
    draw_cards,
)
from .training import CombatTrainingEnv

__all__ = [
    "Action",
    "ACTION_ORDER",
    "CombatState",
    "CombatantState",
    "CombatEnvironment",
    "CombatTrainingEnv",
    "DrawResult",
    "EncounterConfig",
    "InvalidActionError",
    "OBSERVATION_FIELDS",
    "PileState",
    "StepResult",
    "action_to_index",
    "create_initial_combat_state",
    "decode_action_index",
    "draw_cards",
    "enemy_intent_damage",
    "encode_action_mask",
    "encode_observation",
    "legal_action_mask",
]
