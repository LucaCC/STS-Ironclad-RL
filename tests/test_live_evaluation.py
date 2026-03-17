from __future__ import annotations

import json

from sts_ironclad_rl.integration import ActionCommand, GameStateSnapshot
from sts_ironclad_rl.live import (
    ActionDecision,
    EncodedObservation,
    EpisodeFailure,
    EvaluationSummary,
    PolicyEvaluator,
    ReplayEntry,
    RolloutResult,
    format_evaluation_summary,
    summarize_rollouts,
    summary_to_dict,
    summary_to_json,
)


def _observation(*, session_id: str = "session-1") -> EncodedObservation:
    snapshot = GameStateSnapshot(
        session_id=session_id,
        screen_state="COMBAT",
        available_actions=("play_card:0", "end_turn"),
        in_combat=True,
        floor=3,
        act=1,
        raw_state={},
    )
    return EncodedObservation(
        snapshot=snapshot,
        legal_action_ids=snapshot.available_actions,
        features={"turn": 1},
        metadata={},
    )


def _entry(action_id: str, *, step_index: int) -> ReplayEntry:
    return ReplayEntry(
        session_id="session-1",
        step_index=step_index,
        observation=_observation(),
        action=ActionDecision(action_id=action_id),
        command=ActionCommand(session_id="session-1", command=action_id),
    )


def test_summarize_rollouts_aggregates_outcomes_actions_and_proxies() -> None:
    results = (
        RolloutResult(
            session_id="session-1",
            entries=(
                _entry("play_card:0", step_index=0),
                _entry("end_turn", step_index=1),
            ),
            terminal=True,
            step_count=2,
            outcome="victory",
            total_reward=5.0,
            metadata={"final_score": 42, "final_floor": 5},
        ),
        RolloutResult(
            session_id="session-2",
            entries=(_entry("end_turn", step_index=0),),
            terminal=False,
            step_count=1,
            outcome="interrupted",
            failure=EpisodeFailure(
                kind="bridge_disconnect",
                message="timed out",
                step_index=1,
            ),
            total_reward=1.0,
            metadata={"final_floor": 4},
        ),
    )

    summary = summarize_rollouts(results=results, policy_name="heuristic", case_name="smoke")

    assert summary == EvaluationSummary(
        policy_name="heuristic",
        case_name="smoke",
        episode_count=2,
        terminal_episode_count=1,
        interruption_count=1,
        outcome_counts={"failure:bridge_disconnect": 1, "victory": 1},
        failure_counts={"bridge_disconnect": 1},
        action_counts={"end_turn": 2, "play_card:0": 1},
        mean_steps=1.5,
        mean_total_reward=3.0,
        mean_final_score=42.0,
        mean_final_floor=4.5,
        metadata={"terminal_rate": 0.5},
    )


def test_summary_serialization_and_text_format_are_stable() -> None:
    summary = EvaluationSummary(
        policy_name="random_legal",
        case_name="batch-a",
        episode_count=3,
        terminal_episode_count=2,
        interruption_count=1,
        outcome_counts={"combat_end": 1, "victory": 2},
        failure_counts={"max_steps_exceeded": 1},
        action_counts={"end_turn": 4, "play_card:0": 2},
        mean_steps=7.25,
        mean_total_reward=None,
        mean_final_score=12.0,
        mean_final_floor=3.0,
        metadata={"terminal_rate": 2 / 3},
    )

    assert summary_to_dict(summary)["policy_name"] == "random_legal"
    payload = json.loads(summary_to_json(summary))
    assert payload["action_counts"] == {"end_turn": 4, "play_card:0": 2}

    text = format_evaluation_summary(summary)
    assert "policy=random_legal case=batch-a episodes=3" in text
    assert "outcomes=combat_end:1,victory:2" in text
    assert "failures=max_steps_exceeded:1" in text
    assert "mean_final_score=12.00" in text


def test_policy_evaluator_runs_requested_number_of_episodes() -> None:
    class StubRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_episode(self, *, policy, evaluation_case=None) -> RolloutResult:
            del policy
            del evaluation_case
            self.calls += 1
            return RolloutResult(
                session_id=f"session-{self.calls}",
                entries=(),
                terminal=True,
                step_count=0,
                outcome="combat_end",
            )

    class StubPolicy:
        name = "stub"

        def select_action(self, observation: EncodedObservation) -> ActionDecision:
            del observation
            raise AssertionError("runner stub should not call the policy")

    runner = StubRunner()
    result = PolicyEvaluator(runner=runner).evaluate(policy=StubPolicy(), episode_count=3)

    assert runner.calls == 3
    assert result.summary.episode_count == 3
    assert result.summary.outcome_counts == {"combat_end": 3}
