from __future__ import annotations

from sts_ironclad_rl.live import BridgeObservationEncoder, RawStateObservationEncoder


def test_bridge_observation_encoder_exports_stable_features_and_metadata(
    combat_snapshot_factory,
    card_factory,
    enemy_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        turn=4,
        player={"current_hp": 41, "max_hp": 80, "block": 7, "energy": 2},
        hand=(
            card_factory(name="Bash", has_target=True),
            card_factory(name="Defend"),
        ),
        monsters=(
            enemy_factory(name="Jaw Worm", current_hp=34, intent_base_damage=11),
            enemy_factory(name="Cultist", current_hp=0),
        ),
    )

    observation = BridgeObservationEncoder().encode(snapshot)
    structured = observation.metadata["structured_observation"]

    assert observation.legal_action_ids == ("play_card:0:0", "play_card:1", "end_turn")
    assert observation.features == {
        "schema_version": "live_observation.v1",
        "screen_state": "COMBAT",
        "in_combat": True,
        "floor": 3,
        "act": 1,
        "choice_count": 0,
        "legal_action_count": 2,
        "combat_turn": 4,
        "player_current_hp": 41,
        "player_max_hp": 80,
        "player_block": 7,
        "player_energy": 2,
        "hand_count": 2,
        "enemy_count": 2,
        "targetable_enemy_count": 1,
    }
    assert structured["combat"]["player"]["current_hp"] == 41
    assert structured["combat"]["enemies"][0]["is_targetable"] is True
    assert structured["combat"]["enemies"][1]["is_targetable"] is False


def test_bridge_observation_encoder_handles_missing_or_malformed_combat_state(
    combat_snapshot_factory,
) -> None:
    snapshot = combat_snapshot_factory(
        extra_raw_state={"combat_state": {"player": "ignored", "hand": "bad"}}
    )
    observation = BridgeObservationEncoder().encode(snapshot)

    structured = observation.metadata["structured_observation"]
    assert structured["combat"]["player"]["current_hp"] == 0
    assert structured["combat"]["hand"] == ()
    assert structured["combat"]["enemies"] == ()


def test_raw_state_observation_encoder_can_toggle_snapshot_metadata(
    combat_snapshot_factory,
) -> None:
    snapshot = combat_snapshot_factory(extra_raw_state={"reward": 1.5})

    default_observation = RawStateObservationEncoder().encode(snapshot)
    compact_observation = RawStateObservationEncoder(include_snapshot_metadata=False).encode(
        snapshot
    )

    assert default_observation.features == {
        "combat_state": snapshot.raw_state["combat_state"],
        "reward": 1.5,
    }
    assert default_observation.metadata == {
        "screen_state": "COMBAT",
        "in_combat": True,
        "floor": 3,
        "act": 1,
    }
    assert compact_observation.metadata == {}
