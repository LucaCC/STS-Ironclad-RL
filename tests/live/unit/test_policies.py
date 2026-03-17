from __future__ import annotations

import pytest

from sts_ironclad_rl.live import (
    CommunicationModActionContract,
    RandomLegalPolicy,
    SimpleHeuristicPolicy,
)
from tests.live.factories import encode_snapshot


def test_random_policy_always_selects_legal_action(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        hand=(
            card_factory(name="Strike", has_target=True),
            card_factory(name="Defend"),
        ),
        monsters=(enemy_factory(),),
    )
    observation = encode_snapshot(snapshot)
    policy = RandomLegalPolicy(seed=7)

    for _ in range(20):
        assert policy.select_action(observation).action_id in observation.legal_action_ids


def test_random_policy_is_deterministic_for_a_fixed_seed(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        hand=(card_factory(name="Defend"),),
        monsters=(enemy_factory(),),
    )
    observation = encode_snapshot(snapshot)

    first = RandomLegalPolicy(seed=13)
    second = RandomLegalPolicy(seed=13)

    assert [first.select_action(observation).action_id for _ in range(5)] == [
        second.select_action(observation).action_id for _ in range(5)
    ]


def test_random_policy_rejects_observation_without_legal_actions(
    combat_snapshot_factory,
) -> None:
    observation = encode_snapshot(combat_snapshot_factory(available_actions=()))

    with pytest.raises(ValueError, match="does not expose any legal actions"):
        RandomLegalPolicy(seed=1).select_action(observation)


def test_heuristic_policy_prefers_defense_when_low_hp_and_facing_damage(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        player={"current_hp": 20, "max_hp": 80, "block": 0, "energy": 1},
        hand=(
            card_factory(name="Strike", has_target=True),
            card_factory(name="Defend"),
        ),
        monsters=(enemy_factory(intent_base_damage=10),),
    )

    assert (
        SimpleHeuristicPolicy().select_action(encode_snapshot(snapshot)).action_id == "play_card:1"
    )


def test_heuristic_policy_targets_lowest_hp_enemy_with_attack(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        hand=(card_factory(name="Strike", has_target=True),),
        monsters=(
            enemy_factory(name="Louse", current_hp=14, max_hp=14),
            enemy_factory(name="Cultist", current_hp=9, max_hp=48),
        ),
    )

    assert (
        SimpleHeuristicPolicy().select_action(encode_snapshot(snapshot)).action_id
        == "play_card:0:1"
    )


def test_heuristic_policy_prefers_proceed_then_first_choice_outside_combat(
    event_snapshot_factory,
) -> None:
    proceed_observation = encode_snapshot(event_snapshot_factory())
    choose_only_observation = encode_snapshot(event_snapshot_factory(available_actions=("choose",)))

    policy = SimpleHeuristicPolicy()

    assert policy.select_action(proceed_observation).action_id == "proceed"
    assert policy.select_action(choose_only_observation).action_id == "choose:0"


def test_heuristic_policy_ends_turn_when_no_productive_action_exists(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        player={"current_hp": 70, "max_hp": 80, "block": 0, "energy": 0},
        hand=(
            card_factory(name="Strike", is_playable=False, has_target=True),
            card_factory(name="Defend", is_playable=False),
        ),
        monsters=(enemy_factory(current_hp=20, max_hp=20),),
    )

    assert SimpleHeuristicPolicy().select_action(encode_snapshot(snapshot)).action_id == "end_turn"


def test_heuristic_decisions_stay_legal_under_action_contract(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        hand=(
            card_factory(name="Bash", has_target=True),
            card_factory(name="Defend"),
        ),
        monsters=(
            enemy_factory(name="Jaw Worm", current_hp=28, max_hp=28),
            enemy_factory(name="Louse", current_hp=12, max_hp=12),
        ),
    )
    observation = encode_snapshot(snapshot)
    decision = SimpleHeuristicPolicy().select_action(observation)

    command = CommunicationModActionContract().to_validated_command(snapshot, decision)

    assert decision.action_id in observation.legal_action_ids
    assert command.command in {"play", "end", "choose", "proceed", "leave"}
