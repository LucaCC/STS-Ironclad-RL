import pytest

from sts_ironclad_rl.integration import ActionCommand, GameStateSnapshot
from sts_ironclad_rl.live import (
    ActionDecision,
    CommunicationModActionContract,
    RawStateObservationEncoder,
    ReplayEntry,
    ReplayFailure,
)


def test_action_decision_from_action_uses_stable_identifier() -> None:
    decision = ActionDecision.from_action(
        action=CommunicationModActionContract().to_action("end_turn")
    )

    assert decision.action_id == "end_turn"
    assert decision.metadata == {}


def test_raw_state_encoder_uses_live_action_contract_for_legal_ids() -> None:
    snapshot = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("play", "end"),
        in_combat=True,
        floor=1,
        act=1,
        raw_state={
            "combat_state": {
                "hand": [{"is_playable": True, "has_target": False}],
                "monsters": [{"current_hp": 14, "is_gone": False}],
            }
        },
    )

    observation = RawStateObservationEncoder().encode(snapshot)

    assert observation.legal_action_ids == ("play_card:0", "end_turn")
    assert observation.metadata == {
        "screen_state": "COMBAT",
        "in_combat": True,
        "floor": 1,
        "act": 1,
    }


def test_replay_entry_fills_legal_actions_from_observation() -> None:
    snapshot = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("play", "end"),
        in_combat=True,
        raw_state={
            "combat_state": {
                "hand": [{"is_playable": True, "has_target": False}],
                "monsters": [{"current_hp": 14, "is_gone": False}],
            }
        },
    )

    observation = RawStateObservationEncoder().encode(snapshot)
    entry = ReplayEntry(
        session_id="session-1",
        step_index=4,
        observation=observation,
    )

    assert entry.legal_action_ids == ("play_card:0", "end_turn")


def test_replay_entry_rejects_mismatched_session_or_legal_actions() -> None:
    snapshot = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("end",),
        in_combat=True,
        raw_state={},
    )
    observation = RawStateObservationEncoder().encode(snapshot)

    with pytest.raises(ValueError, match="session_id"):
        ReplayEntry(
            session_id="other-session",
            step_index=0,
            observation=observation,
        )

    with pytest.raises(ValueError, match="legal_action_ids"):
        ReplayEntry(
            session_id="session-1",
            step_index=0,
            observation=observation,
            legal_action_ids=("choose:0",),
        )

    with pytest.raises(ValueError, match="command session_id"):
        ReplayEntry(
            session_id="session-1",
            step_index=0,
            observation=observation,
            command=ActionCommand(session_id="other-session", command="end"),
        )


def test_replay_failure_round_trips_through_dict() -> None:
    failure = ReplayFailure(
        stage="bridge_receive",
        message="timed out waiting for state",
        error_type="TimeoutError",
        recoverable=True,
        details={"timeout_seconds": 5.0},
    )

    assert ReplayFailure.from_dict(failure.to_dict()) == failure
