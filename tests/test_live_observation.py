from __future__ import annotations

from sts_ironclad_rl.integration import GameStateSnapshot
from sts_ironclad_rl.live import (
    SCHEMA_VERSION,
    BridgeObservationEncoder,
    ObservationLayout,
    vector_schema,
)


def make_snapshot(
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
        floor=7,
        act=1,
        raw_state=raw_state or {},
    )


def test_bridge_observation_encoder_normalizes_representative_combat_state() -> None:
    encoder = BridgeObservationEncoder(layout=ObservationLayout(max_hand_cards=3, max_enemies=3))
    snapshot = make_snapshot(
        raw_state={
            "screen_type": "NONE",
            "room_phase": "COMBAT",
            "action_phase": "WAITING_ON_USER",
            "gold": 99,
            "ascension_level": 10,
            "class": "IRONCLAD",
            "combat_state": {
                "turn": 3,
                "player": {"current_hp": 51, "max_hp": 80, "block": 12, "energy": 2},
                "hand": [
                    {
                        "id": "strike",
                        "name": "Strike",
                        "cost": 1,
                        "is_playable": True,
                        "has_target": True,
                    },
                    {
                        "id": "defend",
                        "name": "Defend",
                        "cost": 1,
                        "is_playable": True,
                        "has_target": False,
                        "upgraded": True,
                    },
                ],
                "draw_pile": [{}, {}, {}],
                "discard_pile": [{}],
                "exhaust_pile": [],
                "monsters": [
                    {
                        "name": "Jaw Worm",
                        "id": "jaw_worm",
                        "current_hp": 38,
                        "max_hp": 42,
                        "block": 0,
                        "intent": "ATTACK",
                        "move_id": 3,
                        "intent_base_damage": 11,
                        "intent_hits": 1,
                        "is_gone": False,
                    },
                    {
                        "name": "Cultist",
                        "id": "cultist",
                        "current_hp": 48,
                        "max_hp": 48,
                        "block": 6,
                        "intent": "BUFF",
                        "move_id": 1,
                        "intent_base_damage": 0,
                        "intent_hits": 0,
                        "is_gone": False,
                    },
                ],
            },
        }
    )

    observation = encoder.parse(snapshot)
    encoded = encoder.encode(snapshot)

    assert observation.schema_version == SCHEMA_VERSION
    assert observation.combat is not None
    assert observation.combat.turn == 3
    assert observation.combat.player.energy == 2
    assert observation.combat.enemy_count == 2
    assert observation.combat.targetable_enemy_count == 2
    assert encoded.legal_action_ids == ("play_card:0:0", "play_card:0:1", "play_card:1", "end_turn")
    assert encoded.features["player_current_hp"] == 51
    assert encoded.features["hand_0_has_target"] is True
    assert encoded.features["enemy_1_block"] == 6
    assert encoded.metadata["available_actions"] == ("play", "end")


def test_bridge_observation_encoder_defaults_missing_or_partial_fields() -> None:
    encoder = BridgeObservationEncoder(layout=ObservationLayout(max_hand_cards=2, max_enemies=2))
    snapshot = make_snapshot(
        available_actions=("choose",),
        in_combat=False,
        raw_state={
            "choice_list": ["take", 7, None],
            "gold": None,
            "combat_state": {
                "player": {"current_hp": None, "energy": 1},
                "hand": [{}, "bad-card"],
                "monsters": [{"current_hp": None, "is_gone": False}, "bad-monster"],
            },
        },
    )

    observation = encoder.parse(snapshot)
    encoded = encoder.encode(snapshot)

    assert observation.choice_list == ("take",)
    assert observation.gold is None
    assert observation.combat is not None
    assert observation.combat.player.current_hp == 0
    assert observation.combat.player.energy == 1
    assert observation.combat.hand[0].cost == -1
    assert observation.combat.enemies[0].current_hp == 0
    assert observation.combat.enemies[0].is_targetable is False
    assert encoded.legal_action_ids == ("choose:0",)
    assert encoded.features["gold"] == -1
    assert encoded.features["hand_1_present"] is False


def test_bridge_observation_encoder_tracks_targetable_enemies_and_dead_slots() -> None:
    encoder = BridgeObservationEncoder(layout=ObservationLayout(max_hand_cards=2, max_enemies=4))
    snapshot = make_snapshot(
        raw_state={
            "combat_state": {
                "hand": [{"is_playable": True, "has_target": True}],
                "monsters": [
                    {"current_hp": 0, "max_hp": 10, "is_gone": False},
                    {"current_hp": 15, "max_hp": 15, "is_gone": True},
                    {"current_hp": 22, "max_hp": 22, "is_gone": False},
                    {"current_hp": 9, "max_hp": 9, "is_gone": False, "half_dead": True},
                ],
            }
        }
    )

    observation = encoder.parse(snapshot)
    encoded = encoder.encode(snapshot)

    assert observation.combat is not None
    assert observation.combat.targetable_enemy_count == 1
    assert tuple(enemy.is_targetable for enemy in observation.combat.enemies) == (
        False,
        False,
        True,
        False,
    )
    assert encoded.legal_action_ids == ("play_card:0:2", "end_turn")
    assert encoded.features["enemy_2_is_targetable"] is True
    assert encoded.features["enemy_3_half_dead"] is True


def test_bridge_observation_encoder_flat_keys_and_vector_shape_are_stable() -> None:
    layout = ObservationLayout(max_hand_cards=3, max_enemies=2)
    encoder = BridgeObservationEncoder(layout=layout)
    populated = make_snapshot(
        raw_state={
            "combat_state": {
                "turn": 1,
                "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3},
                "hand": [{"cost": 1, "is_playable": True, "has_target": False}],
                "monsters": [{"current_hp": 30, "max_hp": 30, "is_gone": False}],
            }
        }
    )
    sparse = make_snapshot(available_actions=(), in_combat=False, raw_state={})

    populated_obs = encoder.parse(populated)
    sparse_obs = encoder.parse(sparse)
    populated_encoded = encoder.encode(populated)

    assert tuple(populated_obs.flat_dict(layout)) == tuple(sparse_obs.flat_dict(layout))
    assert tuple(populated_obs.flat_dict(layout)) == (
        "schema_version",
        "screen_state",
        "screen_type",
        "room_phase",
        "action_phase",
        "in_combat",
        "floor",
        "act",
        "ascension_level",
        "gold",
        "choice_count",
        "legal_action_count",
        "combat_present",
        "combat_turn",
        "player_current_hp",
        "player_max_hp",
        "player_block",
        "player_energy",
        "draw_pile_size",
        "discard_pile_size",
        "exhaust_pile_size",
        "enemy_count",
        "targetable_enemy_count",
        "hand_count",
        "playable_card_count",
        "targeted_card_count",
        "hand_0_present",
        "hand_0_cost",
        "hand_0_is_playable",
        "hand_0_has_target",
        "hand_0_upgraded",
        "hand_0_exhausts",
        "hand_0_ethereal",
        "hand_1_present",
        "hand_1_cost",
        "hand_1_is_playable",
        "hand_1_has_target",
        "hand_1_upgraded",
        "hand_1_exhausts",
        "hand_1_ethereal",
        "hand_2_present",
        "hand_2_cost",
        "hand_2_is_playable",
        "hand_2_has_target",
        "hand_2_upgraded",
        "hand_2_exhausts",
        "hand_2_ethereal",
        "enemy_0_present",
        "enemy_0_current_hp",
        "enemy_0_max_hp",
        "enemy_0_block",
        "enemy_0_intent_base_damage",
        "enemy_0_intent_hits",
        "enemy_0_is_gone",
        "enemy_0_half_dead",
        "enemy_0_is_targetable",
        "enemy_1_present",
        "enemy_1_current_hp",
        "enemy_1_max_hp",
        "enemy_1_block",
        "enemy_1_intent_base_damage",
        "enemy_1_intent_hits",
        "enemy_1_is_gone",
        "enemy_1_half_dead",
        "enemy_1_is_targetable",
    )
    assert populated_encoded.metadata["vector_schema"] == vector_schema(layout)
    assert len(populated_obs.vector(layout)) == layout.vector_length()
    assert len(sparse_obs.vector(layout)) == layout.vector_length()


def test_bridge_observation_encoder_truncates_only_flat_and_vector_exports() -> None:
    layout = ObservationLayout(max_hand_cards=2, max_enemies=2)
    encoder = BridgeObservationEncoder(layout=layout)
    snapshot = make_snapshot(
        raw_state={
            "combat_state": {
                "hand": [
                    {"cost": 0, "is_playable": True, "has_target": False},
                    {"cost": 1, "is_playable": True, "has_target": False},
                    {"cost": 2, "is_playable": False, "has_target": False},
                ],
                "monsters": [
                    {"current_hp": 20, "max_hp": 20, "is_gone": False},
                    {"current_hp": 15, "max_hp": 15, "is_gone": False},
                    {"current_hp": 10, "max_hp": 10, "is_gone": False},
                ],
            }
        }
    )

    observation = encoder.parse(snapshot)
    flat = observation.flat_dict(layout)

    assert observation.combat is not None
    assert len(observation.combat.hand) == 3
    assert len(observation.combat.enemies) == 3
    assert "hand_2_present" not in flat
    assert "enemy_2_present" not in flat
