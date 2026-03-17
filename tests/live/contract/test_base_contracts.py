from __future__ import annotations

import pytest

from sts_ironclad_rl.integration import ActionCommand, GameStateSnapshot
from sts_ironclad_rl.live import (
    ActionDecision,
    EncodedObservation,
    RawStateObservationEncoder,
    SnapshotActionContract,
)


def test_raw_state_observation_encoder_preserves_snapshot_and_actions() -> None:
    snapshot = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("play_0", "end_turn"),
        in_combat=True,
        floor=7,
        act=1,
        raw_state={"player_hp": 80, "energy": 3},
    )

    encoded = RawStateObservationEncoder().encode(snapshot)

    assert encoded == EncodedObservation(
        snapshot=snapshot,
        legal_action_ids=("play_0", "end_turn"),
        features={"player_hp": 80, "energy": 3},
        metadata={
            "screen_state": "COMBAT",
            "in_combat": True,
            "floor": 7,
            "act": 1,
        },
    )


def test_snapshot_action_contract_maps_decision_to_action_command() -> None:
    contract = SnapshotActionContract()
    decision = ActionDecision(action_id="end_turn", arguments={"source": "policy"})

    command = contract.to_command("session-9", decision)

    assert command == ActionCommand(
        session_id="session-9",
        command="end_turn",
        arguments={"source": "policy"},
    )


def test_snapshot_action_contract_rejects_illegal_actions() -> None:
    snapshot = GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT",
        available_actions=("play_0", "end_turn"),
        in_combat=True,
        raw_state={},
    )
    contract = SnapshotActionContract()

    with pytest.raises(ValueError, match="illegal action"):
        contract.to_validated_command(snapshot, ActionDecision(action_id="play_9"))
