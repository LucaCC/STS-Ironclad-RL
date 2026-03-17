from __future__ import annotations

from sts_ironclad_rl.integration import GameStateSnapshot
from sts_ironclad_rl.live import (
    BridgeObservationEncoder,
    CommunicationModActionContract,
    RandomLegalPolicy,
    SimpleHeuristicPolicy,
)


def _make_snapshot(
    *,
    available_actions: tuple[str, ...] = ("play", "end"),
    in_combat: bool = True,
    raw_state: dict[str, object] | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        session_id="session-1",
        screen_state="COMBAT" if in_combat else "EVENT",
        available_actions=available_actions,
        in_combat=in_combat,
        floor=3,
        act=1,
        raw_state=raw_state or {},
    )


def _encode(snapshot: GameStateSnapshot):
    encoder = BridgeObservationEncoder()
    return encoder.encode(snapshot)


def test_random_policy_always_selects_legal_action() -> None:
    snapshot = _make_snapshot(
        raw_state={
            "combat_state": {
                "hand": [
                    {"name": "Strike", "is_playable": True, "has_target": True},
                    {"name": "Defend", "is_playable": True, "has_target": False},
                ],
                "monsters": [{"current_hp": 18, "max_hp": 18, "is_gone": False}],
            }
        }
    )
    observation = _encode(snapshot)
    policy = RandomLegalPolicy(seed=7)

    for _ in range(20):
        decision = policy.select_action(observation)
        assert decision.action_id in observation.legal_action_ids


def test_random_policy_is_deterministic_for_a_fixed_seed() -> None:
    snapshot = _make_snapshot(
        raw_state={
            "combat_state": {
                "hand": [{"name": "Defend", "is_playable": True, "has_target": False}],
                "monsters": [{"current_hp": 18, "max_hp": 18, "is_gone": False}],
            }
        }
    )
    observation = _encode(snapshot)

    first = RandomLegalPolicy(seed=13)
    second = RandomLegalPolicy(seed=13)

    assert [first.select_action(observation).action_id for _ in range(5)] == [
        second.select_action(observation).action_id for _ in range(5)
    ]


def test_heuristic_policy_prefers_defense_when_low_hp_and_facing_damage() -> None:
    snapshot = _make_snapshot(
        raw_state={
            "combat_state": {
                "player": {"current_hp": 20, "max_hp": 80, "block": 0, "energy": 1},
                "hand": [
                    {"name": "Strike", "is_playable": True, "has_target": True},
                    {"name": "Defend", "is_playable": True, "has_target": False},
                ],
                "monsters": [
                    {
                        "current_hp": 18,
                        "max_hp": 18,
                        "intent_base_damage": 10,
                        "intent_hits": 1,
                        "is_gone": False,
                    }
                ],
            }
        }
    )
    observation = _encode(snapshot)

    assert SimpleHeuristicPolicy().select_action(observation).action_id == "play_card:1"


def test_heuristic_policy_targets_lowest_hp_enemy_with_attack() -> None:
    snapshot = _make_snapshot(
        raw_state={
            "combat_state": {
                "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 2},
                "hand": [{"name": "Strike", "is_playable": True, "has_target": True}],
                "monsters": [
                    {"name": "Louse", "current_hp": 14, "max_hp": 14, "is_gone": False},
                    {"name": "Cultist", "current_hp": 9, "max_hp": 48, "is_gone": False},
                ],
            }
        }
    )
    observation = _encode(snapshot)

    assert SimpleHeuristicPolicy().select_action(observation).action_id == "play_card:0:1"


def test_heuristic_policy_is_deterministic_for_the_same_observation() -> None:
    snapshot = _make_snapshot(
        raw_state={
            "combat_state": {
                "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 2},
                "hand": [{"name": "Strike", "is_playable": True, "has_target": True}],
                "monsters": [
                    {"name": "Jaw Worm", "current_hp": 11, "max_hp": 40, "is_gone": False},
                    {"name": "Cultist", "current_hp": 17, "max_hp": 48, "is_gone": False},
                ],
            }
        }
    )
    observation = _encode(snapshot)
    policy = SimpleHeuristicPolicy()

    assert [policy.select_action(observation).action_id for _ in range(3)] == [
        "play_card:0:0",
        "play_card:0:0",
        "play_card:0:0",
    ]


def test_heuristic_policy_ends_turn_when_no_productive_action_exists() -> None:
    snapshot = _make_snapshot(
        raw_state={
            "combat_state": {
                "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 0},
                "hand": [
                    {"name": "Strike", "is_playable": False, "has_target": True},
                    {"name": "Defend", "is_playable": False, "has_target": False},
                ],
                "monsters": [{"current_hp": 20, "max_hp": 20, "is_gone": False}],
            }
        }
    )
    observation = _encode(snapshot)

    assert SimpleHeuristicPolicy().select_action(observation).action_id == "end_turn"


def test_heuristic_policy_prefers_proceed_outside_combat() -> None:
    snapshot = _make_snapshot(
        available_actions=("choose", "proceed", "leave"),
        in_combat=False,
        raw_state={"choice_list": ["take", "skip"]},
    )
    observation = _encode(snapshot)

    assert SimpleHeuristicPolicy().select_action(observation).action_id == "proceed"


def test_heuristic_policy_falls_back_to_first_choice_when_needed() -> None:
    snapshot = _make_snapshot(
        available_actions=("choose",),
        in_combat=False,
        raw_state={"choice_list": ["take", "skip"]},
    )
    observation = _encode(snapshot)

    assert SimpleHeuristicPolicy().select_action(observation).action_id == "choose:0"


def test_heuristic_decisions_stay_legal_under_action_contract() -> None:
    contract = CommunicationModActionContract()
    snapshot = _make_snapshot(
        raw_state={
            "combat_state": {
                "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 2},
                "hand": [
                    {"name": "Bash", "is_playable": True, "has_target": True},
                    {"name": "Defend", "is_playable": True, "has_target": False},
                ],
                "monsters": [
                    {"current_hp": 28, "max_hp": 28, "is_gone": False},
                    {"current_hp": 12, "max_hp": 12, "is_gone": False},
                ],
            }
        }
    )
    observation = _encode(snapshot)
    decision = SimpleHeuristicPolicy().select_action(observation)

    command = contract.to_validated_command(snapshot, decision)

    assert decision.action_id in observation.legal_action_ids
    assert command.command in {"play", "end", "choose", "proceed", "leave"}
