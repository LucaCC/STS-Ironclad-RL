"""Concrete bridge-backed rollout runner for live episodes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..integration import GameStateSnapshot, LiveGameBridge
from .contracts import (
    ActionContract,
    ActionDecision,
    ActionTraceEntry,
    EncodedObservation,
    EpisodeFailure,
    EvaluationCase,
    ObservationEncoder,
    Policy,
    PolicyContext,
    ReplayEntry,
    ReplaySink,
    RolloutResult,
)


class BridgeDisconnectError(RuntimeError):
    """Raised when the bridge stops yielding snapshots."""


class MalformedStateError(ValueError):
    """Raised when a snapshot or observation violates runner expectations."""


class PolicyOutputError(TypeError):
    """Raised when a policy emits an invalid decision."""


@dataclass(frozen=True)
class RunnerConfig:
    """Guardrails for one live rollout episode."""

    max_steps: int = 200
    max_empty_polls_per_step: int = 5
    max_repeated_state_fingerprints: int = 3
    max_repeated_no_ops: int = 3
    no_op_action_ids: tuple[str, ...] = ("wait", "noop")
    close_bridge_on_exit: bool = True


@dataclass
class LiveEpisodeRunner:
    """Drive the observe -> encode -> act -> command loop for one episode."""

    bridge: LiveGameBridge
    observation_encoder: ObservationEncoder
    action_contract: ActionContract
    replay_sink: ReplaySink | None = None
    config: RunnerConfig = field(default_factory=RunnerConfig)

    def run_episode(
        self,
        *,
        policy: Policy,
        evaluation_case: EvaluationCase | None = None,
    ) -> RolloutResult:
        entries: list[ReplayEntry] = []
        action_trace: list[ActionTraceEntry] = []
        total_reward = 0.0
        reward_seen = False
        repeated_state_count = 0
        repeated_no_op_count = 0
        seen_combat = False
        last_state_fingerprint: str | None = None
        session_id = "unavailable"

        try:
            session = self.bridge.connect()
            session_id = session.session_id
            max_steps = self.config.max_steps
            if evaluation_case is not None and evaluation_case.max_steps is not None:
                max_steps = min(max_steps, evaluation_case.max_steps)

            for step_index in range(max_steps):
                snapshot = self._poll_snapshot(step_index)
                terminal_outcome = self._terminal_outcome(snapshot, seen_combat)
                if snapshot.in_combat:
                    seen_combat = True

                observation = self._encode_observation(snapshot, step_index)

                if terminal_outcome is not None:
                    entry = ReplayEntry(
                        session_id=session.session_id,
                        step_index=step_index,
                        observation=observation,
                        terminal=True,
                        metadata={"outcome": terminal_outcome},
                    )
                    self._log_entry(entry, entries)
                    return self._build_result(
                        session_id=session_id,
                        entries=entries,
                        action_trace=action_trace,
                        outcome=terminal_outcome,
                        terminal=True,
                        total_reward=total_reward if reward_seen else None,
                        metadata={"evaluation_case": self._case_name(evaluation_case)},
                    )

                legal_actions = self._resolve_legal_actions(snapshot, observation, step_index)
                fingerprint = self._snapshot_fingerprint(snapshot)
                repeated_state_count = (
                    repeated_state_count + 1 if fingerprint == last_state_fingerprint else 0
                )
                last_state_fingerprint = fingerprint
                if repeated_state_count >= self.config.max_repeated_state_fingerprints:
                    return self._failure_result(
                        session_id=session_id,
                        entries=entries,
                        action_trace=action_trace,
                        step_count=len(action_trace),
                        kind="desync_guard",
                        message="snapshot repeated without progression",
                        step_index=step_index,
                        metadata={"snapshot_fingerprint": fingerprint},
                        total_reward=total_reward if reward_seen else None,
                    )

                context = PolicyContext(
                    session_id=session_id,
                    step_index=step_index,
                    evaluation_case=evaluation_case,
                    metadata={
                        "screen_state": snapshot.screen_state,
                        "seen_combat": seen_combat,
                        "repeated_state_count": repeated_state_count,
                    },
                )
                decision = self._policy_action(
                    policy, observation, legal_actions, context, step_index
                )
                command = self._map_command(snapshot, legal_actions, decision, step_index)

                if decision.action_id.lower() in self.config.no_op_action_ids:
                    repeated_no_op_count += 1
                else:
                    repeated_no_op_count = 0
                if repeated_no_op_count >= self.config.max_repeated_no_ops:
                    return self._failure_result(
                        session_id=session_id,
                        entries=entries,
                        action_trace=action_trace,
                        step_count=len(action_trace),
                        kind="no_op_guard",
                        message="policy emitted repeated no-op actions",
                        step_index=step_index,
                        metadata={"action_id": decision.action_id},
                        total_reward=total_reward if reward_seen else None,
                    )

                self.bridge.send_action(command)

                entry = ReplayEntry(
                    session_id=session.session_id,
                    step_index=step_index,
                    observation=observation,
                    action=decision,
                    terminal=False,
                    metadata={
                        "command": command.command,
                        "command_arguments": dict(command.arguments),
                        "evaluation_case": self._case_name(evaluation_case),
                    },
                )
                self._log_entry(entry, entries)
                action_trace.append(
                    ActionTraceEntry(
                        step_index=step_index,
                        action_id=decision.action_id,
                        legal_action_ids=legal_actions,
                        command=command.command,
                        command_arguments=dict(command.arguments),
                        metadata=dict(decision.metadata),
                    )
                )

                if entry.reward is not None:
                    total_reward += entry.reward
                    reward_seen = True

            return self._failure_result(
                session_id=session_id,
                entries=entries,
                action_trace=action_trace,
                step_count=len(action_trace),
                kind="max_steps_exceeded",
                message="episode exceeded configured step limit",
                step_index=max_steps,
                metadata={"max_steps": max_steps},
                total_reward=total_reward if reward_seen else None,
            )
        except Exception as exc:
            return self._failure_result(
                session_id=session_id,
                entries=entries,
                action_trace=action_trace,
                step_count=len(action_trace),
                kind=self._failure_kind(exc),
                message=str(exc),
                step_index=len(action_trace),
                metadata={"exception_type": type(exc).__name__},
                total_reward=total_reward if reward_seen else None,
            )
        finally:
            if self.config.close_bridge_on_exit:
                self.bridge.close()

    def _poll_snapshot(self, step_index: int) -> GameStateSnapshot:
        for _ in range(self.config.max_empty_polls_per_step):
            self.bridge.request_state()
            snapshot = self.bridge.receive_state()
            if snapshot is not None:
                return snapshot
        msg = "bridge did not provide a game state snapshot"
        raise BridgeDisconnectError(msg)

    def _encode_observation(
        self, snapshot: GameStateSnapshot, step_index: int
    ) -> EncodedObservation:
        observation = self.observation_encoder.encode(snapshot)
        if observation.snapshot.session_id != snapshot.session_id:
            msg = "encoded observation session_id does not match the snapshot"
            raise MalformedStateError(msg)
        if step_index < 0:
            msg = "step_index must be non-negative"
            raise MalformedStateError(msg)
        return observation

    def _resolve_legal_actions(
        self,
        snapshot: GameStateSnapshot,
        observation: EncodedObservation,
        step_index: int,
    ) -> tuple[str, ...]:
        legal_actions = tuple(observation.legal_action_ids) or tuple(
            self.action_contract.legal_action_ids(snapshot)
        )
        if not legal_actions:
            msg = f"no legal actions available at step {step_index}"
            raise MalformedStateError(msg)
        return legal_actions

    def _policy_action(
        self,
        policy: Policy,
        observation: EncodedObservation,
        legal_actions: tuple[str, ...],
        context: PolicyContext,
        step_index: int,
    ) -> ActionDecision:
        decision = policy.act(observation, legal_actions, context)
        if not isinstance(decision, ActionDecision):
            msg = (
                "policy.act must return ActionDecision, "
                f"got {type(decision).__name__} at step {step_index}"
            )
            raise PolicyOutputError(msg)
        if decision.action_id not in legal_actions:
            msg = f"policy selected illegal action: {decision.action_id}"
            raise PolicyOutputError(msg)
        return decision

    def _map_command(
        self,
        snapshot: GameStateSnapshot,
        legal_actions: tuple[str, ...],
        decision: ActionDecision,
        step_index: int,
    ):
        if decision.action_id not in legal_actions:
            msg = f"policy selected illegal action: {decision.action_id}"
            raise PolicyOutputError(msg)

        to_validated_command = getattr(self.action_contract, "to_validated_command", None)
        if callable(to_validated_command):
            return to_validated_command(snapshot, decision)

        command = self.action_contract.to_command(snapshot.session_id, decision)
        if command.session_id != snapshot.session_id:
            msg = f"mapped command session mismatch at step {step_index}"
            raise MalformedStateError(msg)
        return command

    def _log_entry(self, entry: ReplayEntry, entries: list[ReplayEntry]) -> None:
        entries.append(entry)
        if self.replay_sink is not None:
            self.replay_sink.log(entry)

    def _terminal_outcome(self, snapshot: GameStateSnapshot, seen_combat: bool) -> str | None:
        if snapshot.in_combat:
            return None
        if not seen_combat:
            return "not_in_combat"

        raw_state = snapshot.raw_state
        if raw_state.get("victory") is True:
            return "victory"
        if raw_state.get("player_dead") is True:
            return "defeat"
        return "combat_end"

    def _build_result(
        self,
        *,
        session_id: str,
        entries: list[ReplayEntry],
        action_trace: list[ActionTraceEntry],
        outcome: str,
        terminal: bool,
        total_reward: float | None,
        metadata: Mapping[str, Any],
    ) -> RolloutResult:
        return RolloutResult(
            session_id=session_id,
            entries=tuple(entries),
            action_trace=tuple(action_trace),
            terminal=terminal,
            step_count=len(action_trace),
            outcome=outcome,
            total_reward=total_reward,
            metadata=dict(metadata),
        )

    def _failure_result(
        self,
        *,
        session_id: str,
        entries: list[ReplayEntry],
        action_trace: list[ActionTraceEntry],
        step_count: int,
        kind: str,
        message: str,
        step_index: int,
        metadata: Mapping[str, Any],
        total_reward: float | None,
    ) -> RolloutResult:
        failure = EpisodeFailure(
            kind=kind,
            message=message,
            step_index=step_index,
            metadata=dict(metadata),
        )
        return RolloutResult(
            session_id=session_id,
            entries=tuple(entries),
            action_trace=tuple(action_trace),
            terminal=False,
            step_count=step_count,
            outcome="failure",
            failure=failure,
            total_reward=total_reward,
            metadata={"failure_kind": kind},
        )

    def _snapshot_fingerprint(self, snapshot: GameStateSnapshot) -> str:
        payload = {
            "screen_state": snapshot.screen_state,
            "available_actions": snapshot.available_actions,
            "in_combat": snapshot.in_combat,
            "floor": snapshot.floor,
            "act": snapshot.act,
            "raw_state": snapshot.raw_state,
        }
        return json.dumps(payload, sort_keys=True, default=repr)

    def _failure_kind(self, exc: Exception) -> str:
        if isinstance(exc, BridgeDisconnectError):
            return "bridge_disconnect"
        if isinstance(exc, PolicyOutputError):
            return "invalid_policy_output"
        if isinstance(exc, (MalformedStateError, ValueError)):
            return "malformed_state"
        return "runner_error"

    def _case_name(self, evaluation_case: EvaluationCase | None) -> str | None:
        if evaluation_case is None:
            return None
        return evaluation_case.name
