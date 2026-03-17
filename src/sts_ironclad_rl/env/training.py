"""Training-facing wrapper for the milestone 1 combat slice.

This module keeps combat transitions inside ``CombatEnvironment`` and exposes a
thin RL loop interface:

- ``reset(seed=...) -> (observation, info)``
- ``step(action_index) -> (observation, reward, terminated, truncated, info)``

Policy actions are stable integer indices over ``ACTION_ORDER``.
"""

from __future__ import annotations

from random import Random
from typing import Any

from .combat import CombatEnvironment, InvalidActionError
from .encoding import (
    ACTION_ORDER,
    OBSERVATION_FIELDS,
    decode_action_index,
    encode_observation,
    legal_action_mask,
)
from .state import CombatState, EncounterConfig


class CombatTrainingEnv:
    """Thin RL wrapper over the deterministic combat core."""

    WIN_REWARD = 1.0
    LOSS_REWARD = -1.0
    STEP_REWARD = 0.0

    def __init__(
        self,
        config: EncounterConfig,
        *,
        max_steps: int = 100,
        seed: int = 0,
    ) -> None:
        if max_steps <= 0:
            msg = "max_steps must be positive"
            raise ValueError(msg)

        self._combat_env = CombatEnvironment(config)
        self._max_steps = max_steps
        self._seed_rng = Random(seed)
        self._episode_seed: int | None = None
        self._elapsed_steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[tuple[int, ...], dict[str, Any]]:
        """Reset the environment and return the encoded observation plus info."""
        episode_seed = self._seed_rng.randrange(2**32) if seed is None else seed
        state = self._combat_env.reset(seed=episode_seed)
        self._episode_seed = episode_seed
        self._elapsed_steps = 0
        return encode_observation(state), self._info(state)

    def step(self, action_index: int) -> tuple[tuple[int, ...], float, bool, bool, dict[str, Any]]:
        """Apply one indexed policy action and return a training-friendly step tuple."""
        if self._episode_seed is None:
            msg = "environment must be reset before use"
            raise RuntimeError(msg)

        try:
            action = decode_action_index(action_index)
        except ValueError as exc:
            msg = f"unknown policy action index: {action_index}"
            raise InvalidActionError(msg) from exc

        result = self._combat_env.step(action)
        self._elapsed_steps += 1

        terminated = result.done
        truncated = not terminated and self._elapsed_steps >= self._max_steps
        reward = self.STEP_REWARD
        if terminated:
            reward = self.WIN_REWARD if result.state.enemy.hp == 0 else self.LOSS_REWARD
        info = self._info(
            result.state,
            selected_action_index=action_index,
            terminated=terminated,
            truncated=truncated,
        )
        return encode_observation(result.state), reward, terminated, truncated, info

    @staticmethod
    def action_mapping() -> dict[int, str]:
        """Return the stable policy-index to combat-action mapping."""
        return {index: action.value for index, action in enumerate(ACTION_ORDER)}

    def _info(
        self,
        state: CombatState,
        *,
        selected_action_index: int | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> dict[str, Any]:
        action_mask = legal_action_mask(state)
        legal_action_indices = tuple(
            index for index, is_legal in enumerate(action_mask) if is_legal
        )
        info: dict[str, Any] = {
            "seed": self._episode_seed,
            "episode_step": self._elapsed_steps,
            "action_mask": action_mask,
            "legal_action_indices": legal_action_indices,
            "action_mapping": self.action_mapping(),
            "observation_fields": OBSERVATION_FIELDS,
            "combat_state": state,
        }
        if selected_action_index is not None:
            info["selected_action_index"] = selected_action_index
        if terminated:
            info["terminal_reason"] = "victory" if state.enemy.hp == 0 else "defeat"
            info["won"] = state.enemy.hp == 0
            info["final_player_hp"] = state.player.hp
            info["final_enemy_hp"] = state.enemy.hp
        elif truncated:
            info["terminal_reason"] = "max_steps"
        return info
