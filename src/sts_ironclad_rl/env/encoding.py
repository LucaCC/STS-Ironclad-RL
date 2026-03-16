"""Stable action and observation encodings for the first trainable combat slice.

The milestone 1 contract intentionally stays small and mirrors only currently
supported mechanics:

- actions are fixed indices over the supported action set
- masks mark illegal actions explicitly
- observations are flat integer features derived directly from ``CombatState``

This module does not attempt to model broader Slay the Spire state.
"""

from __future__ import annotations

from collections.abc import Mapping

from .combat import Action, enemy_intent_damage
from .state import CombatState

ACTION_ORDER: tuple[Action, ...] = (
    Action.ATTACK,
    Action.DEFEND,
    Action.END_TURN,
)
"""Stable action ordering for milestone 1 training."""

OBSERVATION_FIELDS: tuple[str, ...] = (
    "player_hp",
    "player_max_hp",
    "player_block",
    "energy",
    "turn",
    "enemy_hp",
    "enemy_max_hp",
    "enemy_block",
    "enemy_intent_damage",
    "hand_strike_count",
    "hand_defend_count",
    "draw_strike_count",
    "draw_defend_count",
    "discard_strike_count",
    "discard_defend_count",
    "draw_pile_count",
    "hand_count",
    "discard_pile_count",
    "exhaust_pile_count",
)
"""Field names for the flat observation vector returned by ``encode_observation``."""

_ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}


def action_to_index(action: Action) -> int:
    """Return the stable index for an action."""
    return _ACTION_TO_INDEX[action]


def decode_action_index(action_index: int) -> Action:
    """Decode a stable action index into an environment action."""
    if action_index < 0 or action_index >= len(ACTION_ORDER):
        msg = f"unknown action index: {action_index}"
        raise ValueError(msg)
    return ACTION_ORDER[action_index]


def encode_action_mask(action_mask: Mapping[Action, bool]) -> tuple[bool, ...]:
    """Encode an action mask into stable action-index order."""
    return tuple(action_mask[action] for action in ACTION_ORDER)


def legal_action_mask(state: CombatState) -> tuple[bool, ...]:
    """Return the legal action mask for a combat state in stable index order."""
    has_energy = state.energy > 0
    hand = state.piles.hand
    return (
        has_energy and "strike" in hand,
        has_energy and "defend" in hand,
        True,
    )


def encode_observation(state: CombatState) -> tuple[int, ...]:
    """Encode a combat state into a deterministic flat observation vector."""
    hand = state.piles.hand
    draw_pile = state.piles.draw_pile
    discard_pile = state.piles.discard_pile
    return (
        state.player.hp,
        state.player.max_hp,
        state.player.block,
        state.energy,
        state.turn,
        state.enemy.hp,
        state.enemy.max_hp,
        state.enemy.block,
        enemy_intent_damage(state),
        hand.count("strike"),
        hand.count("defend"),
        draw_pile.count("strike"),
        draw_pile.count("defend"),
        discard_pile.count("strike"),
        discard_pile.count("defend"),
        len(draw_pile),
        len(hand),
        len(discard_pile),
        len(state.piles.exhaust_pile),
    )
