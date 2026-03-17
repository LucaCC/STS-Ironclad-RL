"""Bridge-backed live episode execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..integration import ActionCommand, GameStateSnapshot, LiveGameBridge
from .contracts import (
    ActionContract,
    EpisodeFailure,
    EvaluationCase,
    ObservationEncoder,
    Policy,
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
        session_id = "unavailable"
        total_reward = 0.0
        reward_seen = False
        seen_combat = False

        try:
            session = self.bridge.connect()
            session_id = session.session_id
            max_steps = self._resolve_max_steps(evaluation_case)

            for step_index in range(max_steps):
                snapshot = self._poll_snapshot()
                observation = self.observation_encoder.encode(snapshot)
                self._validate_observation(snapshot, observation.snapshot.session_id)

                terminal_outcome = self._terminal_outcome(snapshot, seen_combat)
                if snapshot.in_combat:
                    seen_combat = True

                reward = _extract_reward(snapshot.raw_state)
                if reward is not None:
                    total_reward += reward
                    reward_seen = True

                if terminal_outcome is not None:
                    entry = ReplayEntry(
                        session_id=session_id,
                        step_index=step_index,
                        observation=observation,
                        reward=reward,
                        terminal=True,
                        outcome=terminal_outcome,
                        metadata={"evaluation_case": _case_name(evaluation_case)},
                    )
                    self._log_entry(entry, entries)
                    return RolloutResult(
                        session_id=session_id,
                        entries=tuple(entries),
                        terminal=True,
                        step_count=sum(
                            1 for replay_entry in entries if replay_entry.action is not None
                        ),
                        outcome=terminal_outcome,
                        total_reward=total_reward if reward_seen else None,
                        metadata=_result_metadata(
                            evaluation_case=evaluation_case,
                            final_snapshot=snapshot,
                        ),
                    )

                legal_action_ids = tuple(observation.legal_action_ids) or tuple(
                    self.action_contract.legal_action_ids(snapshot)
                )
                if not legal_action_ids:
                    raise MalformedStateError(f"no legal actions available at step {step_index}")

                decision = policy.select_action(observation)
                if decision.action_id not in legal_action_ids:
                    raise PolicyOutputError(f"policy selected illegal action: {decision.action_id}")

                command = self._to_command(snapshot, decision)
                self.bridge.send_action(command)

                entry = ReplayEntry(
                    session_id=session_id,
                    step_index=step_index,
                    observation=observation,
                    action=decision,
                    command=command,
                    reward=reward,
                    metadata={"evaluation_case": _case_name(evaluation_case)},
                )
                self._log_entry(entry, entries)

            return self._failure_result(
                session_id=session_id,
                entries=entries,
                kind="max_steps_exceeded",
                message="episode exceeded configured step limit",
                step_index=max_steps,
                evaluation_case=evaluation_case,
                total_reward=total_reward if reward_seen else None,
            )
        except Exception as exc:
            return self._failure_result(
                session_id=session_id,
                entries=entries,
                kind=_failure_kind(exc),
                message=str(exc),
                step_index=len(entries),
                evaluation_case=evaluation_case,
                total_reward=total_reward if reward_seen else None,
            )
        finally:
            if self.config.close_bridge_on_exit:
                self.bridge.close()

    def _resolve_max_steps(self, evaluation_case: EvaluationCase | None) -> int:
        if evaluation_case is None or evaluation_case.max_steps is None:
            return self.config.max_steps
        return min(self.config.max_steps, evaluation_case.max_steps)

    def _poll_snapshot(self) -> GameStateSnapshot:
        for _ in range(self.config.max_empty_polls_per_step):
            self.bridge.request_state()
            snapshot = self.bridge.receive_state()
            if snapshot is not None:
                return snapshot
        raise BridgeDisconnectError("bridge did not provide a game state snapshot")

    def _validate_observation(self, snapshot: GameStateSnapshot, observed_session_id: str) -> None:
        if observed_session_id != snapshot.session_id:
            raise MalformedStateError("encoded observation session_id does not match snapshot")

    def _to_command(self, snapshot: GameStateSnapshot, decision) -> ActionCommand:
        to_validated_command = getattr(self.action_contract, "to_validated_command", None)
        if callable(to_validated_command):
            command = to_validated_command(snapshot, decision)
        else:
            command = self.action_contract.to_command(snapshot.session_id, decision)
        if command.session_id != snapshot.session_id:
            raise MalformedStateError("mapped command session_id does not match snapshot")
        return command

    def _log_entry(self, entry: ReplayEntry, entries: list[ReplayEntry]) -> None:
        entries.append(entry)
        if self.replay_sink is not None:
            self.replay_sink.log(entry)

    def _failure_result(
        self,
        *,
        session_id: str,
        entries: list[ReplayEntry],
        kind: str,
        message: str,
        step_index: int,
        evaluation_case: EvaluationCase | None,
        total_reward: float | None,
    ) -> RolloutResult:
        return RolloutResult(
            session_id=session_id,
            entries=tuple(entries),
            terminal=False,
            step_count=sum(1 for replay_entry in entries if replay_entry.action is not None),
            outcome="interrupted",
            failure=EpisodeFailure(kind=kind, message=message, step_index=step_index),
            total_reward=total_reward,
            metadata={"evaluation_case": _case_name(evaluation_case)},
        )

    def _terminal_outcome(self, snapshot: GameStateSnapshot, seen_combat: bool) -> str | None:
        if snapshot.in_combat:
            return None
        if not seen_combat:
            return None

        raw_state = snapshot.raw_state
        if raw_state.get("victory") is True:
            return "victory"
        if raw_state.get("player_dead") is True:
            return "defeat"
        return "combat_end"


def _failure_kind(exc: Exception) -> str:
    if isinstance(exc, BridgeDisconnectError):
        return "bridge_disconnect"
    if isinstance(exc, PolicyOutputError):
        return "invalid_policy_output"
    if isinstance(exc, MalformedStateError | ValueError):
        return "malformed_state"
    return "runner_error"


def _extract_reward(raw_state: Mapping[str, Any]) -> float | None:
    reward = raw_state.get("reward")
    if isinstance(reward, bool):
        return float(int(reward))
    if isinstance(reward, int | float):
        return float(reward)
    return None


def _result_metadata(
    *,
    evaluation_case: EvaluationCase | None,
    final_snapshot: GameStateSnapshot,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "evaluation_case": _case_name(evaluation_case),
        "final_floor": final_snapshot.floor,
    }
    score = final_snapshot.raw_state.get("score")
    if isinstance(score, bool):
        metadata["final_score"] = int(score)
    elif isinstance(score, int | float):
        metadata["final_score"] = score
    return metadata


def _case_name(evaluation_case: EvaluationCase | None) -> str | None:
    if evaluation_case is None:
        return None
    return evaluation_case.name
