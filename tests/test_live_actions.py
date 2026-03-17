from __future__ import annotations

import pytest

from sts_ironclad_rl.integration import ActionCommand, GameStateSnapshot
from sts_ironclad_rl.live import (
    ActionDecision,
    ChooseAction,
    CommunicationModActionContract,
    EndTurnAction,
    LeaveAction,
    MonsterTarget,
    PlayCardAction,
    ProceedAction,
    action_from_id,
)


def make_snapshot(
    *,
    available_actions: tuple[str, ...],
    in_combat: bool = False,
    raw_state: dict[str, object] | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT" if in_combat else "EVENT",
        available_actions=available_actions,
        in_combat=in_combat,
        raw_state=raw_state or {},
    )


def test_action_construction_round_trips_through_action_ids() -> None:
    assert action_from_id("play_card:0:2") == PlayCardAction(
        hand_index=0,
        target=MonsterTarget(monster_index=2),
    )
    assert action_from_id("play_card:1") == PlayCardAction(hand_index=1)
    assert action_from_id("choose:3") == ChooseAction(choice_index=3)
    assert action_from_id("end_turn") == EndTurnAction()
    assert action_from_id("proceed") == ProceedAction()
    assert action_from_id("leave") == LeaveAction()


def test_command_mapping_uses_bridge_safe_arguments() -> None:
    contract = CommunicationModActionContract()

    assert contract.to_command(
        session_id="session-1",
        decision=ActionDecision.from_action(PlayCardAction(hand_index=0, target=MonsterTarget(1))),
    ) == ActionCommand(
        session_id="session-1",
        command="play",
        arguments={"card_index": 1, "target_index": 1},
    )
    assert contract.to_command(
        session_id="session-1",
        decision=ActionDecision.from_action(ChooseAction(choice_index=2)),
    ) == ActionCommand(
        session_id="session-1",
        command="choose",
        arguments={"choice_index": 2},
    )


def test_invalid_action_rejection_catches_malformed_and_illegal_actions() -> None:
    contract = CommunicationModActionContract()
    snapshot = make_snapshot(
        available_actions=("play", "end"),
        in_combat=True,
        raw_state={
            "combat_state": {
                "hand": [
                    {"is_playable": False, "has_target": False},
                    {"is_playable": True, "has_target": True},
                ],
                "monsters": [{"current_hp": 12, "is_gone": False}],
            }
        },
    )

    with pytest.raises(ValueError, match="unsupported action_id"):
        contract.to_validated_command(snapshot, ActionDecision(action_id="play_card"))

    with pytest.raises(ValueError, match="card is not playable"):
        contract.to_validated_command(snapshot, ActionDecision(action_id="play_card:0"))

    with pytest.raises(ValueError, match="targeted card requires a target"):
        contract.to_validated_command(snapshot, ActionDecision(action_id="play_card:1"))

    with pytest.raises(ValueError, match="action is unavailable"):
        contract.to_validated_command(snapshot, ActionDecision(action_id="leave"))


def test_legality_filtering_returns_only_actions_supported_by_live_state() -> None:
    contract = CommunicationModActionContract()
    snapshot = make_snapshot(
        available_actions=("choose", "play", "end"),
        in_combat=True,
        raw_state={
            "choice_list": ["rest", "smith"],
            "combat_state": {
                "hand": [
                    {"is_playable": True, "has_target": True},
                    {"is_playable": True, "has_target": False},
                    {"is_playable": False, "has_target": False},
                ],
                "monsters": [
                    {"current_hp": 10, "is_gone": False},
                    {"current_hp": 0, "is_gone": False},
                ],
            },
        },
    )

    assert contract.legal_action_ids(snapshot) == (
        "choose:0",
        "choose:1",
        "play_card:0:0",
        "play_card:1",
        "end_turn",
    )


def test_target_resolution_ignores_dead_or_gone_monsters() -> None:
    contract = CommunicationModActionContract()
    snapshot = make_snapshot(
        available_actions=("play",),
        in_combat=True,
        raw_state={
            "combat_state": {
                "hand": [{"is_playable": True, "has_target": True}],
                "monsters": [
                    {"current_hp": 0, "is_gone": False},
                    {"current_hp": 9, "is_gone": True},
                    {"current_hp": 11, "is_gone": False},
                ],
            }
        },
    )

    assert contract.legal_action_ids(snapshot) == ("play_card:0:2",)
    assert contract.to_validated_command(
        snapshot,
        ActionDecision(action_id="play_card:0:2"),
    ) == ActionCommand(
        session_id="session-1",
        command="play",
        arguments={"card_index": 1, "target_index": 2},
    )

    with pytest.raises(ValueError, match="invalid target_index"):
        contract.to_validated_command(snapshot, ActionDecision(action_id="play_card:0:0"))


def test_end_turn_behavior_requires_available_end_command() -> None:
    contract = CommunicationModActionContract()
    combat_snapshot = make_snapshot(available_actions=("end",), in_combat=True)
    noncombat_snapshot = make_snapshot(available_actions=("end",), in_combat=False)

    assert contract.legal_action_ids(combat_snapshot) == ("end_turn",)
    assert contract.to_validated_command(
        combat_snapshot,
        ActionDecision(action_id="end_turn"),
    ) == ActionCommand(
        session_id="session-1",
        command="end",
        arguments={},
    )

    with pytest.raises(ValueError, match="only valid while in combat"):
        contract.to_validated_command(noncombat_snapshot, ActionDecision(action_id="end_turn"))
