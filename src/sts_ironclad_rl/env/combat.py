"""Minimal combat-only environment with deterministic transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .state import (
    CombatantState,
    CombatState,
    EncounterConfig,
    PileState,
    create_initial_combat_state,
    draw_cards,
)


class Action(StrEnum):
    """Minimal action set for the first combat environment slice."""

    ATTACK = "attack"
    DEFEND = "defend"
    END_TURN = "end_turn"


@dataclass(frozen=True)
class StepResult:
    """Result of a single environment transition."""

    state: CombatState
    reward: float
    done: bool


class InvalidActionError(ValueError):
    """Raised when a chosen action is not currently legal."""


class CombatEnvironment:
    """Single-encounter deterministic combat environment."""

    def __init__(self, config: EncounterConfig) -> None:
        self._config = config
        self._state: CombatState | None = None

    def reset(self, seed: int) -> CombatState:
        """Reset the encounter from a deterministic seed."""
        self._state = create_initial_combat_state(seed=seed, config=self._config)
        return self._state

    def action_mask(self) -> dict[Action, bool]:
        """Return which actions are legal in the current state."""
        state = self._require_state()
        return {
            Action.ATTACK: state.energy > 0 and "strike" in state.piles.hand,
            Action.DEFEND: state.energy > 0 and "defend" in state.piles.hand,
            Action.END_TURN: True,
        }

    def step(self, action: Action) -> StepResult:
        """Apply one action and advance the combat deterministically."""
        state = self._require_state()
        if not self.action_mask()[action]:
            msg = f"illegal action: {action}"
            raise InvalidActionError(msg)

        next_state = state
        reward = 0.0

        if action is Action.ATTACK:
            next_state = _play_card(next_state, card_id="strike", block_gain=0, enemy_damage=6)
            reward = 6.0
        elif action is Action.DEFEND:
            next_state = _play_card(next_state, card_id="defend", block_gain=5, enemy_damage=0)
        else:
            next_state = _end_turn(next_state)

        done = next_state.enemy.hp == 0 or next_state.player.hp == 0
        self._state = next_state
        return StepResult(state=next_state, reward=reward, done=done)

    def _require_state(self) -> CombatState:
        if self._state is None:
            msg = "environment must be reset before use"
            raise RuntimeError(msg)
        return self._state


def _play_card(state: CombatState, card_id: str, block_gain: int, enemy_damage: int) -> CombatState:
    hand = list(state.piles.hand)
    hand.remove(card_id)

    next_enemy_hp = max(0, state.enemy.hp - enemy_damage)
    next_player = CombatantState(
        hp=state.player.hp,
        max_hp=state.player.max_hp,
        block=state.player.block + block_gain,
    )
    next_enemy = CombatantState(
        hp=next_enemy_hp,
        max_hp=state.enemy.max_hp,
        block=state.enemy.block,
    )
    next_piles = PileState(
        draw_pile=state.piles.draw_pile,
        hand=tuple(hand),
        discard_pile=state.piles.discard_pile + (card_id,),
        exhaust_pile=state.piles.exhaust_pile,
    )
    return CombatState(
        seed=state.seed,
        turn=state.turn,
        energy=state.energy - 1,
        draw_per_turn=state.draw_per_turn,
        player=next_player,
        enemy=next_enemy,
        piles=next_piles,
    )


def _end_turn(state: CombatState) -> CombatState:
    post_enemy = _apply_enemy_turn(state)
    refreshed_piles = _discard_hand(post_enemy.piles)
    start_turn = CombatState(
        seed=post_enemy.seed,
        turn=post_enemy.turn + 1,
        energy=3,
        draw_per_turn=post_enemy.draw_per_turn,
        player=CombatantState(
            hp=post_enemy.player.hp,
            max_hp=post_enemy.player.max_hp,
            block=0,
        ),
        enemy=post_enemy.enemy,
        piles=refreshed_piles,
    )
    return draw_cards(start_turn, start_turn.draw_per_turn).state


def _apply_enemy_turn(state: CombatState) -> CombatState:
    damage = _enemy_damage(seed=state.seed, turn=state.turn)
    prevented = min(state.player.block, damage)
    next_player = CombatantState(
        hp=max(0, state.player.hp - (damage - prevented)),
        max_hp=state.player.max_hp,
        block=state.player.block - prevented,
    )
    return CombatState(
        seed=state.seed,
        turn=state.turn,
        energy=state.energy,
        draw_per_turn=state.draw_per_turn,
        player=next_player,
        enemy=state.enemy,
        piles=state.piles,
    )


def _discard_hand(piles: PileState) -> PileState:
    draw_pile = piles.draw_pile
    discard_pile = piles.discard_pile + piles.hand
    if not draw_pile and discard_pile:
        draw_pile = discard_pile
        discard_pile = ()
    return PileState(
        draw_pile=draw_pile,
        hand=(),
        discard_pile=discard_pile,
        exhaust_pile=piles.exhaust_pile,
    )


def _enemy_damage(seed: int, turn: int) -> int:
    return 6 + ((seed + turn) % 2)
