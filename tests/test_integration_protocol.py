from sts_ironclad_rl.integration import (
    ActionCommand,
    BridgeEnvelope,
    BridgeMessageType,
    GameStateSnapshot,
    TrajectoryEntry,
)


def test_bridge_envelope_from_message_serializes_dataclasses() -> None:
    snapshot = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("play_strike_0", "end_turn"),
        in_combat=True,
        floor=1,
        act=1,
        raw_state={"turn": 1},
    )

    envelope = BridgeEnvelope.from_message(BridgeMessageType.GAME_STATE, snapshot)

    assert envelope.to_dict() == {
        "message_type": "game_state",
        "payload": {
            "session_id": "session-1",
            "screen_state": "COMBAT",
            "available_actions": ("play_strike_0", "end_turn"),
            "in_combat": True,
            "floor": 1,
            "act": 1,
            "raw_state": {"turn": 1},
        },
    }


def test_trajectory_entry_keeps_observation_and_action_linked() -> None:
    observation = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("end_turn",),
        in_combat=True,
    )
    action = ActionCommand(session_id="session-1", command="end_turn")
    entry = TrajectoryEntry(
        session_id="session-1",
        step_index=3,
        observation=observation,
        action=action,
        reward=0.0,
        terminal=True,
    )

    assert entry.observation.session_id == entry.action.session_id == entry.session_id
    assert entry.terminal is True
