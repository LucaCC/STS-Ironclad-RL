"""Minimal live-game RL contracts built on top of the bridge package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..integration import ActionCommand, GameStateSnapshot


@dataclass(frozen=True)
class EncodedObservation:
    """Policy-facing representation derived from one live game snapshot."""

    snapshot: GameStateSnapshot
    legal_action_ids: tuple[str, ...]
    features: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDecision:
    """Policy-selected action in the repo-level live action namespace."""

    action_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayEntry:
    """Structured replay record for one live rollout step."""

    session_id: str
    step_index: int
    observation: EncodedObservation
    action: ActionDecision | None = None
    command: ActionCommand | None = None
    reward: float | None = None
    terminal: bool = False
    outcome: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeFailure:
    """Structured failure or interruption information for one episode."""

    kind: str
    message: str
    step_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutResult:
    """Outcome of a single live rollout or bounded session slice."""

    session_id: str
    entries: tuple[ReplayEntry, ...]
    terminal: bool
    step_count: int
    outcome: str | None = None
    failure: EpisodeFailure | None = None
    total_reward: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationCase:
    """Named evaluation scenario for a live rollout batch."""

    name: str
    max_steps: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation output built from rollout results."""

    policy_name: str
    case_name: str
    episode_count: int
    terminal_episode_count: int
    interruption_count: int
    outcome_counts: Mapping[str, int]
    failure_counts: Mapping[str, int]
    action_counts: Mapping[str, int]
    mean_steps: float
    mean_total_reward: float | None = None
    mean_final_score: float | None = None
    mean_final_floor: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObservationEncoder(Protocol):
    """Translate raw bridge snapshots into policy-facing observations."""

    def encode(self, snapshot: GameStateSnapshot) -> EncodedObservation:
        """Encode a live snapshot for policy consumption."""


class ActionContract(Protocol):
    """Own the policy-facing action namespace and bridge mapping."""

    def legal_action_ids(self, snapshot: GameStateSnapshot) -> tuple[str, ...]:
        """Return the legal policy-facing actions for a snapshot."""

    def to_command(self, session_id: str, decision: ActionDecision) -> ActionCommand:
        """Translate one policy decision into a bridge action command."""


class ReplaySink(Protocol):
    """Write structured replay records for later analysis."""

    def log(self, entry: ReplayEntry) -> None:
        """Persist one replay entry."""


class Policy(Protocol):
    """Minimal policy contract for live-game decisions."""

    name: str

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        """Choose one action for the provided encoded observation."""


class RolloutRunner(Protocol):
    """Run the bridge-facing control loop for one policy."""

    def run_episode(
        self,
        *,
        policy: Policy,
        evaluation_case: EvaluationCase | None = None,
    ) -> RolloutResult:
        """Run one live episode or bounded session slice."""


@dataclass(frozen=True)
class RawStateObservationEncoder:
    """Conservative default encoder that exposes raw bridge state directly."""

    include_snapshot_metadata: bool = True

    def encode(self, snapshot: GameStateSnapshot) -> EncodedObservation:
        metadata: dict[str, Any] = {}
        if self.include_snapshot_metadata:
            metadata = {
                "screen_state": snapshot.screen_state,
                "in_combat": snapshot.in_combat,
                "floor": snapshot.floor,
                "act": snapshot.act,
            }

        return EncodedObservation(
            snapshot=snapshot,
            legal_action_ids=snapshot.available_actions,
            features=dict(snapshot.raw_state),
            metadata=metadata,
        )


@dataclass(frozen=True)
class SnapshotActionContract:
    """Minimal action contract that mirrors the bridge's available actions."""

    def legal_action_ids(self, snapshot: GameStateSnapshot) -> tuple[str, ...]:
        return snapshot.available_actions

    def to_command(self, session_id: str, decision: ActionDecision) -> ActionCommand:
        return ActionCommand(
            session_id=session_id,
            command=decision.action_id,
            arguments=dict(decision.arguments),
        )

    def validate(self, snapshot: GameStateSnapshot, decision: ActionDecision) -> None:
        """Reject actions that are not exposed by the current snapshot."""
        if decision.action_id not in snapshot.available_actions:
            msg = f"illegal action for snapshot: {decision.action_id}"
            raise ValueError(msg)

    def to_validated_command(
        self,
        snapshot: GameStateSnapshot,
        decision: ActionDecision,
    ) -> ActionCommand:
        """Validate against the snapshot before producing a bridge command."""
        self.validate(snapshot, decision)
        return self.to_command(snapshot.session_id, decision)
