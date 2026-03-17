from __future__ import annotations

import pytest

from sts_ironclad_rl.live import (
    ActionDecision,
    ChooseAction,
    CommunicationModActionContract,
    EndTurnAction,
    MonsterTarget,
    PlayCardAction,
    ProceedAction,
    action_from_id,
    action_to_id,
)


def test_canonical_actions_round_trip_through_action_ids() -> None:
    actions = (
        PlayCardAction(hand_index=0),
        PlayCardAction(hand_index=1, target=MonsterTarget(2)),
        EndTurnAction(),
        ChooseAction(choice_index=1),
        ProceedAction(),
    )

    assert tuple(action_from_id(action_to_id(action)) for action in actions) == actions


def test_communication_mod_action_contract_expands_legal_actions(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        available_actions=("choose", "play", "end"),
        hand=(
            card_factory(name="Strike", has_target=True),
            card_factory(name="Defend"),
            card_factory(name="Wound", is_playable=False),
        ),
        monsters=(
            enemy_factory(name="Jaw Worm", current_hp=40),
            enemy_factory(name="Cultist", current_hp=0),
            enemy_factory(name="Louse", current_hp=11),
        ),
        extra_raw_state={"choice_list": ["take", "skip"]},
    )

    legal_action_ids = CommunicationModActionContract().legal_action_ids(snapshot)

    assert legal_action_ids == (
        "choose:0",
        "choose:1",
        "play_card:0:0",
        "play_card:0:2",
        "play_card:1",
        "end_turn",
    )


def test_communication_mod_action_contract_validates_target_requirements(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        hand=(card_factory(name="Strike", has_target=True),),
        monsters=(enemy_factory(name="Cultist", current_hp=16),),
    )
    contract = CommunicationModActionContract()

    with pytest.raises(ValueError, match="requires a target"):
        contract.to_validated_command(snapshot, ActionDecision(action_id="play_card:0"))

    with pytest.raises(ValueError, match="invalid target_index"):
        contract.to_validated_command(snapshot, ActionDecision(action_id="play_card:0:3"))


def test_communication_mod_action_contract_rejects_free_form_arguments(
    combat_snapshot_factory,
) -> None:
    snapshot = combat_snapshot_factory()

    with pytest.raises(ValueError, match="do not accept free-form"):
        CommunicationModActionContract().to_validated_command(
            snapshot,
            ActionDecision(action_id="end_turn", arguments={"ignored": True}),
        )
