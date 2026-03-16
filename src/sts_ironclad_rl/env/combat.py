"""Deterministic combat environment built around card-play actions."""

from __future__ import annotations

from dataclasses import dataclass

from .state import (
    CombatState,
    EncounterConfig,
    MonsterState,
    PileState,
    PlayerState,
    create_initial_combat_state,
    draw_cards,
)


@dataclass(frozen=True)
class PlayCardAction:
    """Play the card at the given hand index, optionally targeting a monster."""

    card_index: int
    target_index: int | None = None


@dataclass(frozen=True)
class EndTurnAction:
    """End the current player turn and resolve enemy actions."""


CombatAction = PlayCardAction | EndTurnAction


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

    @property
    def state(self) -> CombatState:
        """Return the current combat state."""
        return self._require_state()

    def legal_actions(self) -> tuple[CombatAction, ...]:
        """Return the currently legal structured combat actions."""
        return legal_actions(self._require_state())

    def action_mask(self) -> dict[CombatAction, bool]:
        """Return a convenience mapping of legal actions for the current state."""
        return {action: True for action in self.legal_actions()}

    def step(self, action: CombatAction) -> StepResult:
        """Apply one action and advance the combat deterministically."""
        state = self._require_state()
        result = apply_action(state, action)
        self._state = result.state
        return result

    def _require_state(self) -> CombatState:
        if self._state is None:
            msg = "environment must be reset before use"
            raise RuntimeError(msg)
        return self._state


def legal_actions(state: CombatState) -> tuple[CombatAction, ...]:
    """Return all legal combat actions for the current state."""
    if state.is_terminal():
        return ()

    living_targets = tuple(
        index for index, monster in enumerate(state.monsters) if monster.is_alive
    )
    actions: list[CombatAction] = []
    for card_index, card in enumerate(state.piles.hand):
        if card.cost > state.player.energy:
            continue
        if card.requires_target:
            for target_index in living_targets:
                actions.append(PlayCardAction(card_index=card_index, target_index=target_index))
        else:
            actions.append(PlayCardAction(card_index=card_index))

    actions.append(EndTurnAction())
    return tuple(actions)


def apply_action(state: CombatState, action: CombatAction) -> StepResult:
    """Apply one combat action to a state and return the next deterministic result."""
    if action not in legal_actions(state):
        msg = f"illegal action: {action}"
        raise InvalidActionError(msg)

    if isinstance(action, PlayCardAction):
        next_state, reward = _play_card(state, action)
    else:
        next_state, reward = _end_turn(state)

    return StepResult(state=next_state, reward=reward, done=next_state.is_terminal())


def _play_card(state: CombatState, action: PlayCardAction) -> tuple[CombatState, float]:
    card = state.piles.hand[action.card_index]
    if card.requires_target and action.target_index is None:
        msg = "targeted cards require a target_index"
        raise InvalidActionError(msg)

    hand = list(state.piles.hand)
    played_card = hand.pop(action.card_index)
    discard_pile = list(state.piles.discard_pile)
    exhaust_pile = list(state.piles.exhaust_pile)
    if played_card.exhausts:
        exhaust_pile.append(played_card)
    else:
        discard_pile.append(played_card)

    monsters = list(state.monsters)
    damage_dealt = 0
    if played_card.damage > 0:
        target_index = action.target_index
        if target_index is None or not 0 <= target_index < len(monsters):
            msg = "invalid target_index for damage card"
            raise InvalidActionError(msg)
        target = monsters[target_index]
        if not target.is_alive:
            msg = "cannot target a defeated monster"
            raise InvalidActionError(msg)
        next_target, damage_dealt = _apply_damage(target, played_card.damage)
        monsters[target_index] = next_target

    player = PlayerState(
        hp=state.player.hp,
        max_hp=state.player.max_hp,
        block=state.player.block + played_card.block,
        energy=state.player.energy - played_card.cost,
    )
    next_state = CombatState(
        seed=state.seed,
        turn=state.turn,
        starting_energy=state.starting_energy,
        draw_per_turn=state.draw_per_turn,
        shuffle_count=state.shuffle_count,
        player=player,
        monsters=tuple(monsters),
        piles=PileState(
            draw_pile=state.piles.draw_pile,
            hand=tuple(hand),
            discard_pile=tuple(discard_pile),
            exhaust_pile=tuple(exhaust_pile),
        ),
    )
    return next_state, float(damage_dealt)


def _end_turn(state: CombatState) -> tuple[CombatState, float]:
    discarded_hand = state.piles.discard_pile + state.piles.hand
    end_turn_state = CombatState(
        seed=state.seed,
        turn=state.turn,
        starting_energy=state.starting_energy,
        draw_per_turn=state.draw_per_turn,
        shuffle_count=state.shuffle_count,
        player=state.player,
        monsters=state.monsters,
        piles=PileState(
            draw_pile=state.piles.draw_pile,
            hand=(),
            discard_pile=discarded_hand,
            exhaust_pile=state.piles.exhaust_pile,
        ),
    )
    post_enemy_state, damage_taken = _apply_enemy_turn(end_turn_state)
    next_turn_state = CombatState(
        seed=post_enemy_state.seed,
        turn=post_enemy_state.turn + 1,
        starting_energy=post_enemy_state.starting_energy,
        draw_per_turn=post_enemy_state.draw_per_turn,
        shuffle_count=post_enemy_state.shuffle_count,
        player=PlayerState(
            hp=post_enemy_state.player.hp,
            max_hp=post_enemy_state.player.max_hp,
            block=0,
            energy=post_enemy_state.starting_energy,
        ),
        monsters=_advance_monster_intents(post_enemy_state.monsters),
        piles=post_enemy_state.piles,
    )
    drawn_state = draw_cards(next_turn_state, next_turn_state.draw_per_turn).state
    return drawn_state, float(-damage_taken)


def _apply_enemy_turn(state: CombatState) -> tuple[CombatState, int]:
    monsters: list[MonsterState] = []
    player = state.player
    total_damage_taken = 0

    for monster in state.monsters:
        if not monster.is_alive:
            monsters.append(monster)
            continue

        updated_monster = monster
        if monster.intent.block:
            updated_monster = MonsterState(
                monster_id=monster.monster_id,
                name=monster.name,
                hp=monster.hp,
                max_hp=monster.max_hp,
                block=monster.block + monster.intent.block,
                intent_cycle=monster.intent_cycle,
                intent_index=monster.intent_index,
            )

        pending_damage = monster.intent.damage * monster.intent.hits
        if pending_damage:
            remaining_block = max(0, player.block - pending_damage)
            prevented = player.block - remaining_block
            hp_loss = pending_damage - prevented
            player = PlayerState(
                hp=max(0, player.hp - hp_loss),
                max_hp=player.max_hp,
                block=remaining_block,
                energy=player.energy,
            )
            total_damage_taken += hp_loss

        monsters.append(updated_monster)

    return (
        CombatState(
            seed=state.seed,
            turn=state.turn,
            starting_energy=state.starting_energy,
            draw_per_turn=state.draw_per_turn,
            shuffle_count=state.shuffle_count,
            player=player,
            monsters=tuple(monsters),
            piles=state.piles,
        ),
        total_damage_taken,
    )


def _advance_monster_intents(monsters: tuple[MonsterState, ...]) -> tuple[MonsterState, ...]:
    advanced_monsters: list[MonsterState] = []
    for monster in monsters:
        next_index = (monster.intent_index + 1) % len(monster.intent_cycle)
        advanced_monsters.append(
            MonsterState(
                monster_id=monster.monster_id,
                name=monster.name,
                hp=monster.hp,
                max_hp=monster.max_hp,
                block=monster.block,
                intent_cycle=monster.intent_cycle,
                intent_index=next_index,
            )
        )
    return tuple(advanced_monsters)


def _apply_damage(monster: MonsterState, damage: int) -> tuple[MonsterState, int]:
    prevented = min(monster.block, damage)
    hp_loss = damage - prevented
    next_monster = MonsterState(
        monster_id=monster.monster_id,
        name=monster.name,
        hp=max(0, monster.hp - hp_loss),
        max_hp=monster.max_hp,
        block=monster.block - prevented,
        intent_cycle=monster.intent_cycle,
        intent_index=monster.intent_index,
    )
    return next_monster, hp_loss
