"""Simple baseline policies for deterministic combat benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from random import Random

from ..env import Action, CombatState
from ..env.combat import enemy_intent_damage

LETHAL_ATTACK_DAMAGE = 6
DEFEND_BLOCK_GAIN = 5


def legal_actions(action_mask: Mapping[Action, bool]) -> tuple[Action, ...]:
    """Return the currently legal actions in stable enum order."""
    return tuple(action for action in Action if action_mask[action])


@dataclass
class BaselinePolicy:
    """Base class for simple policies that choose from the legal action mask."""

    name: str

    def choose_action(self, state: CombatState, action_mask: Mapping[Action, bool]) -> Action:
        """Choose one legal action for the provided combat state."""
        raise NotImplementedError

    def select_action(
        self,
        state: CombatState,
        action_mask: Mapping[Action, bool],
        rng: Random,
    ) -> Action:
        """Compatibility adapter for trainer and evaluator code."""
        del rng
        return self.choose_action(state, action_mask)


@dataclass
class RandomPolicy(BaselinePolicy):
    """Uniformly sample from the currently legal actions."""

    rng: Random = field(default_factory=Random)
    name: str = "random"

    def choose_action(self, state: CombatState, action_mask: Mapping[Action, bool]) -> Action:
        del state
        actions = legal_actions(action_mask)
        if not actions:
            msg = "action mask must contain at least one legal action"
            raise ValueError(msg)
        return self.rng.choice(actions)

    def select_action(
        self,
        state: CombatState,
        action_mask: Mapping[Action, bool],
        rng: Random,
    ) -> Action:
        del state
        actions = legal_actions(action_mask)
        if not actions:
            msg = "action mask must contain at least one legal action"
            raise ValueError(msg)
        return rng.choice(actions)


@dataclass
class HeuristicPolicy(BaselinePolicy):
    """Deterministic combat heuristic for sanity-check benchmarking."""

    danger_threshold: int = 6
    name: str = "heuristic"

    def choose_action(self, state: CombatState, action_mask: Mapping[Action, bool]) -> Action:
        actions = legal_actions(action_mask)
        if not actions:
            msg = "action mask must contain at least one legal action"
            raise ValueError(msg)

        if action_mask[Action.ATTACK] and state.enemy.hp <= LETHAL_ATTACK_DAMAGE:
            return Action.ATTACK

        if action_mask[Action.DEFEND]:
            incoming_damage = max(0, enemy_intent_damage(state) - state.player.block)
            prevented_damage = min(incoming_damage, DEFEND_BLOCK_GAIN)
            if incoming_damage >= self.danger_threshold and prevented_damage > 0:
                return Action.DEFEND

        if action_mask[Action.ATTACK]:
            return Action.ATTACK

        if action_mask[Action.DEFEND]:
            return Action.DEFEND

        return Action.END_TURN
