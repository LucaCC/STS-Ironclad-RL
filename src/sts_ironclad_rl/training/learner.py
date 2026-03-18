"""Learner-facing contracts built on top of the shared live rollout path."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from ..integration import GameStateSnapshot
from ..live import (
    ActionDecision,
    BridgeObservationEncoder,
    EncodedObservation,
    EndTurnAction,
    LiveObservation,
    MonsterTarget,
    ObservationLayout,
    PlayCardAction,
    ReplayEntry,
    action_from_id,
)

LEARNER_OBSERVATION_SCHEMA_VERSION = "learner_observation.v1"
LEARNER_TRANSITION_SCHEMA_VERSION = "learner_transition.v1"
PAD_TOKEN_VALUE = 0.0

ObservationSource: TypeAlias = (
    GameStateSnapshot | EncodedObservation | ReplayEntry | LiveObservation
)


@dataclass(frozen=True)
class LearnerObservationLayout:
    """Fixed observation layout consumed by learning code."""

    max_hand_slots: int = ObservationLayout().max_hand_cards
    max_enemies: int = ObservationLayout().max_enemies

    @property
    def player_feature_names(self) -> tuple[str, ...]:
        return ("player_hp", "player_max_hp", "player_block", "player_energy")

    @property
    def combat_context_feature_names(self) -> tuple[str, ...]:
        return ("draw_pile_size", "discard_pile_size", "exhaust_pile_size", "turn_index")

    @property
    def hand_slot_feature_names(self) -> tuple[str, ...]:
        return ("card_token", "cost", "is_playable", "has_target")

    @property
    def enemy_slot_feature_names(self) -> tuple[str, ...]:
        return ("current_hp", "max_hp", "block", "intent_token", "alive", "targetable")

    @property
    def vector_size(self) -> int:
        return len(self.feature_names())

    def feature_names(self) -> tuple[str, ...]:
        names = list(self.player_feature_names)
        names.extend(self.combat_context_feature_names)
        names.extend(f"hand_mask_{slot}" for slot in range(self.max_hand_slots))
        for slot in range(self.max_hand_slots):
            names.extend(
                f"hand_{slot}_{feature_name}" for feature_name in self.hand_slot_feature_names
            )
        names.extend(f"enemy_mask_{slot}" for slot in range(self.max_enemies))
        for slot in range(self.max_enemies):
            names.extend(
                f"enemy_{slot}_{feature_name}" for feature_name in self.enemy_slot_feature_names
            )
        return tuple(names)


@dataclass(frozen=True)
class LearnerObservation:
    """Fixed-size learner observation vector plus section masks."""

    schema_version: str
    vector: tuple[float, ...]
    hand_mask: tuple[int, ...]
    enemy_mask: tuple[int, ...]


@dataclass(frozen=True)
class LearnerActionIndex:
    """Stable discrete action indexing for combat-only learning."""

    max_hand_slots: int = ObservationLayout().max_hand_cards
    max_enemies: int = ObservationLayout().max_enemies

    @property
    def size(self) -> int:
        return 1 + self.max_hand_slots + (self.max_hand_slots * self.max_enemies)

    def describe(self, action_index: int) -> str:
        if action_index == 0:
            return "END_TURN"

        hand_start = 1
        targeted_start = hand_start + self.max_hand_slots
        if hand_start <= action_index < targeted_start:
            return f"PLAY_CARD_HAND_{action_index - hand_start}_NO_TARGET"
        if targeted_start <= action_index < self.size:
            offset = action_index - targeted_start
            hand_index, enemy_index = divmod(offset, self.max_enemies)
            return f"PLAY_CARD_HAND_{hand_index}_TARGET_ENEMY_{enemy_index}"

        msg = f"action_index out of range: {action_index}"
        raise ValueError(msg)

    def index_to_action_id(self, action_index: int) -> str:
        if action_index == 0:
            return EndTurnAction().action_id

        hand_start = 1
        targeted_start = hand_start + self.max_hand_slots
        if hand_start <= action_index < targeted_start:
            return PlayCardAction(hand_index=action_index - hand_start).action_id
        if targeted_start <= action_index < self.size:
            offset = action_index - targeted_start
            hand_index, enemy_index = divmod(offset, self.max_enemies)
            return PlayCardAction(
                hand_index=hand_index,
                target=MonsterTarget(enemy_index),
            ).action_id

        msg = f"action_index out of range: {action_index}"
        raise ValueError(msg)

    def action_to_index(self, action: str | ActionDecision) -> int:
        action_id = action.action_id if isinstance(action, ActionDecision) else action
        parsed_action = action_from_id(action_id)

        if isinstance(parsed_action, EndTurnAction):
            return 0
        if not isinstance(parsed_action, PlayCardAction):
            msg = f"unsupported learner action: {action_id}"
            raise ValueError(msg)
        if parsed_action.hand_index >= self.max_hand_slots:
            msg = f"hand_index exceeds learner bounds: {parsed_action.hand_index}"
            raise ValueError(msg)
        if parsed_action.target is None:
            return 1 + parsed_action.hand_index
        if parsed_action.target.monster_index >= self.max_enemies:
            msg = f"monster_index exceeds learner bounds: {parsed_action.target.monster_index}"
            raise ValueError(msg)
        return (
            1
            + self.max_hand_slots
            + (parsed_action.hand_index * self.max_enemies)
            + parsed_action.target.monster_index
        )

    def legal_mask(self, source: ObservationSource) -> tuple[int, ...]:
        if isinstance(source, LiveObservation):
            msg = "legal_mask requires a snapshot-backed source"
            raise TypeError(msg)

        snapshot = _snapshot_from_source(source)
        mask = [0] * self.size
        if snapshot.in_combat:
            mask[0] = 1

        for action_id in BridgeObservationEncoder().action_contract.legal_action_ids(snapshot):
            try:
                mask[self.action_to_index(action_id)] = 1
            except ValueError:
                continue
        return tuple(mask)


@dataclass(frozen=True)
class RewardConfig:
    """Configurable shaped reward weights for learner transitions."""

    enemy_hp_weight: float = 0.1
    player_hp_weight: float = 0.2
    terminal_win_bonus: float = 1.0
    terminal_loss_penalty: float = 1.0
    step_penalty: float = 0.01


@dataclass(frozen=True)
class LearnerRewardFunction:
    """Compute shaped combat rewards from adjacent live observations."""

    config: RewardConfig = field(default_factory=RewardConfig)
    observation_encoder: "LearnerObservationEncoder" = field(
        default_factory=lambda: LearnerObservationEncoder()
    )

    def reward(
        self,
        current: ObservationSource,
        nxt: ObservationSource,
        *,
        done: bool,
        outcome: str | None,
    ) -> float:
        current_live = self.observation_encoder.live_observation(current)
        next_live = self.observation_encoder.live_observation(nxt)

        next_enemy_hp = _resolved_next_enemy_hp(
            current_observation=current_live,
            next_observation=next_live,
            done=done,
            outcome=outcome,
        )
        next_player_hp = _resolved_next_player_hp(
            current_observation=current_live,
            next_observation=next_live,
            done=done,
            outcome=outcome,
        )
        enemy_hp_delta = max(0, _enemy_hp_total(current_live) - next_enemy_hp)
        player_hp_delta = max(0, _player_hp(current_live) - next_player_hp)

        reward = (enemy_hp_delta * self.config.enemy_hp_weight) - (
            player_hp_delta * self.config.player_hp_weight
        )
        reward -= self.config.step_penalty

        if done and outcome == "victory":
            reward += self.config.terminal_win_bonus
        elif done and outcome == "defeat":
            reward -= self.config.terminal_loss_penalty

        return reward


@dataclass(frozen=True)
class LearnerTransition:
    """Replay-derived learner transition tuple for masked DQN training."""

    schema_version: str
    state: tuple[float, ...]
    action_index: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    mask: tuple[int, ...]

    def as_tuple(
        self,
    ) -> tuple[tuple[float, ...], int, float, tuple[float, ...], bool, tuple[int, ...]]:
        return (self.state, self.action_index, self.reward, self.next_state, self.done, self.mask)


@dataclass(frozen=True)
class LearnerObservationEncoder:
    """Convert live observations into a fixed learner vector."""

    layout: LearnerObservationLayout = field(default_factory=LearnerObservationLayout)
    bridge_encoder: BridgeObservationEncoder = field(default_factory=BridgeObservationEncoder)

    def encode(self, source: ObservationSource) -> LearnerObservation:
        live_observation = self.live_observation(source)
        snapshot = (
            _snapshot_from_source(source) if not isinstance(source, LiveObservation) else None
        )
        combat_state = {}
        if snapshot is not None:
            raw_combat_state = snapshot.raw_state.get("combat_state")
            if isinstance(raw_combat_state, dict):
                combat_state = raw_combat_state

        combat = live_observation.combat
        player_values = (
            float(0 if combat is None else combat.player.current_hp),
            float(0 if combat is None else combat.player.max_hp),
            float(0 if combat is None else combat.player.block),
            float(0 if combat is None else combat.player.energy),
        )
        context_values = (
            float(_pile_size(combat_state, "draw_pile")),
            float(_pile_size(combat_state, "discard_pile")),
            float(_pile_size(combat_state, "exhaust_pile")),
            float(0 if combat is None else combat.turn),
        )

        hand = () if combat is None else combat.hand[: self.layout.max_hand_slots]
        hand_mask = tuple(
            1 if slot < len(hand) else 0 for slot in range(self.layout.max_hand_slots)
        )
        hand_features: list[float] = []
        for slot in range(self.layout.max_hand_slots):
            if slot < len(hand):
                card = hand[slot]
                hand_features.extend(
                    (
                        _stable_token_value(card.card_id or card.name),
                        float(card.cost),
                        float(int(card.is_playable)),
                        float(int(card.has_target)),
                    )
                )
            else:
                hand_features.extend((PAD_TOKEN_VALUE, 0.0, 0.0, 0.0))

        enemies = () if combat is None else combat.enemies[: self.layout.max_enemies]
        enemy_mask = tuple(
            1 if slot < len(enemies) else 0 for slot in range(self.layout.max_enemies)
        )
        enemy_features: list[float] = []
        for slot in range(self.layout.max_enemies):
            if slot < len(enemies):
                enemy = enemies[slot]
                alive = int(enemy.current_hp > 0 and not enemy.is_gone and not enemy.half_dead)
                enemy_features.extend(
                    (
                        float(enemy.current_hp),
                        float(enemy.max_hp),
                        float(enemy.block),
                        _stable_token_value(enemy.intent),
                        float(alive),
                        float(int(enemy.is_targetable)),
                    )
                )
            else:
                enemy_features.extend((0.0, 0.0, 0.0, PAD_TOKEN_VALUE, 0.0, 0.0))

        vector = tuple(
            player_values
            + context_values
            + tuple(float(value) for value in hand_mask)
            + tuple(hand_features)
            + tuple(float(value) for value in enemy_mask)
            + tuple(enemy_features)
        )
        return LearnerObservation(
            schema_version=LEARNER_OBSERVATION_SCHEMA_VERSION,
            vector=vector,
            hand_mask=hand_mask,
            enemy_mask=enemy_mask,
        )

    def live_observation(self, source: ObservationSource) -> LiveObservation:
        if isinstance(source, LiveObservation):
            return source
        return self.bridge_encoder.parse(_snapshot_from_source(source))


@dataclass(frozen=True)
class LearnerTransitionExtractor:
    """Build learner transitions from the shared replay records."""

    observation_encoder: LearnerObservationEncoder = field(
        default_factory=LearnerObservationEncoder
    )
    action_index: LearnerActionIndex = field(default_factory=LearnerActionIndex)
    reward_function: LearnerRewardFunction = field(default_factory=LearnerRewardFunction)

    def extract(self, entries: Sequence[ReplayEntry]) -> tuple[LearnerTransition, ...]:
        transitions: list[LearnerTransition] = []
        for current_entry, next_entry in zip(entries, entries[1:]):
            if current_entry.action is None:
                continue

            state = self.observation_encoder.encode(current_entry.observation).vector
            next_state = self.observation_encoder.encode(next_entry.observation).vector
            done = bool(next_entry.terminal)
            transitions.append(
                LearnerTransition(
                    schema_version=LEARNER_TRANSITION_SCHEMA_VERSION,
                    state=state,
                    action_index=self.action_index.action_to_index(current_entry.action),
                    reward=self.reward_function.reward(
                        current_entry.observation,
                        next_entry.observation,
                        done=done,
                        outcome=next_entry.outcome,
                    ),
                    next_state=next_state,
                    done=done,
                    mask=self.action_index.legal_mask(next_entry.observation),
                )
            )
        return tuple(transitions)

    def extract_from_rollouts(
        self,
        episodes: Iterable[Sequence[ReplayEntry]],
    ) -> tuple[LearnerTransition, ...]:
        transitions: list[LearnerTransition] = []
        for entries in episodes:
            transitions.extend(self.extract(tuple(entries)))
        return tuple(transitions)


def _snapshot_from_source(
    source: GameStateSnapshot | EncodedObservation | ReplayEntry,
) -> GameStateSnapshot:
    if isinstance(source, GameStateSnapshot):
        return source
    if isinstance(source, EncodedObservation):
        return source.snapshot
    if isinstance(source, ReplayEntry):
        return source.observation.snapshot
    msg = f"unsupported observation source: {type(source).__name__}"
    raise TypeError(msg)


def _pile_size(combat_state: dict[str, object], pile_name: str) -> int:
    explicit_size = combat_state.get(f"{pile_name}_size")
    if isinstance(explicit_size, bool):
        return int(explicit_size)
    if isinstance(explicit_size, int):
        return explicit_size

    pile = combat_state.get(pile_name)
    if isinstance(pile, list):
        return len(pile)
    return 0


def _stable_token_value(token: str | None) -> float:
    if not token:
        return PAD_TOKEN_VALUE
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16))


def _enemy_hp_total(observation: LiveObservation) -> int:
    if observation.combat is None:
        return 0
    return sum(
        max(0, enemy.current_hp) for enemy in observation.combat.enemies if not enemy.is_gone
    )


def _player_hp(observation: LiveObservation) -> int:
    if observation.combat is None:
        return 0
    return max(0, observation.combat.player.current_hp)


def _resolved_next_enemy_hp(
    *,
    current_observation: LiveObservation,
    next_observation: LiveObservation,
    done: bool,
    outcome: str | None,
) -> int:
    if next_observation.combat is not None or not done:
        return _enemy_hp_total(next_observation)
    if outcome in {"victory", "combat_end"}:
        return 0
    return _enemy_hp_total(current_observation)


def _resolved_next_player_hp(
    *,
    current_observation: LiveObservation,
    next_observation: LiveObservation,
    done: bool,
    outcome: str | None,
) -> int:
    if next_observation.combat is not None or not done:
        return _player_hp(next_observation)
    if outcome == "defeat":
        return 0
    return _player_hp(current_observation)
