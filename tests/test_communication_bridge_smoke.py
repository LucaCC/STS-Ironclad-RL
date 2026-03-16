from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "communication_bridge_smoke.py"
    spec = importlib.util.spec_from_file_location("communication_bridge_smoke", module_path)
    if spec is None or spec.loader is None:
        msg = "failed to load communication_bridge_smoke module"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


communication_bridge_smoke = _load_module()
append_log = communication_bridge_smoke.append_log
build_run_info = communication_bridge_smoke.build_run_info
choose_command = communication_bridge_smoke.choose_command
choose_choice_command = communication_bridge_smoke.choose_choice_command
choose_combat_command = communication_bridge_smoke.choose_combat_command
choose_confirmation_command = communication_bridge_smoke.choose_confirmation_command
choose_in_game_command = communication_bridge_smoke.choose_in_game_command
choose_play_command = communication_bridge_smoke.choose_play_command
choose_shop_command = communication_bridge_smoke.choose_shop_command
extract_run_seed = communication_bridge_smoke.extract_run_seed
first_affordable_index = communication_bridge_smoke.first_affordable_index
first_living_monster_index = communication_bridge_smoke.first_living_monster_index
fingerprint_message = communication_bridge_smoke.fingerprint_message
persist_run_artifacts = communication_bridge_smoke.persist_run_artifacts
summarize_message = communication_bridge_smoke.summarize_message


def test_choose_command_prefers_safe_noop_in_run() -> None:
    message = {
        "ready_for_command": True,
        "in_game": True,
        "available_commands": ["wait", "state"],
        "game_state": {},
    }

    assert choose_command(message) == "WAIT 30"


def test_choose_command_autostarts_when_start_is_available() -> None:
    message = {
        "ready_for_command": True,
        "in_game": False,
        "available_commands": ["start", "state"],
    }

    assert choose_command(message) == "START IRONCLAD"


def test_choose_command_requests_state_outside_run_after_autostart() -> None:
    message = {"ready_for_command": True, "in_game": False, "available_commands": ["state"]}

    assert choose_command(message, has_started_run=True) == "STATE"


def test_choose_choice_command_uses_first_named_choice_when_simple() -> None:
    game_state = {"choice_list": ["x=0", "x=4"]}

    assert choose_choice_command(game_state, {"choose"}) == "CHOOSE x=0"


def test_choose_choice_command_falls_back_to_first_index_for_spaced_names() -> None:
    game_state = {"choice_list": ["Gain a rare card", "Remove a card"]}

    assert choose_choice_command(game_state, {"choose"}) == "CHOOSE 0"


def test_choose_command_prefers_first_choice_in_game() -> None:
    message = {
        "ready_for_command": True,
        "in_game": True,
        "available_commands": ["choose", "wait", "state"],
        "game_state": {"choice_list": ["x=0", "x=4"]},
    }

    assert choose_command(message) == "CHOOSE x=0"


def test_choose_confirmation_command_proceeds_after_choice() -> None:
    game_state = {"screen_type": "GRID"}

    assert (
        choose_confirmation_command(
            game_state,
            {"proceed", "state"},
            last_command="CHOOSE 0",
        )
        == "PROCEED"
    )


def test_choose_confirmation_command_skips_map_and_shop_room() -> None:
    assert (
        choose_confirmation_command(
            {"screen_type": "MAP"},
            {"proceed", "state"},
            last_command="CHOOSE x=0",
        )
        is None
    )
    assert (
        choose_confirmation_command(
            {"screen_type": "SHOP_ROOM"},
            {"proceed", "state"},
            last_command="CHOOSE shop",
        )
        is None
    )


def test_choose_shop_command_enters_shop_only_when_enabled() -> None:
    game_state = {
        "screen_type": "SHOP_ROOM",
        "choice_list": ["shop"],
    }

    assert (
        choose_shop_command(game_state, {"choose", "proceed"}, want_to_shop=True) == "CHOOSE shop"
    )
    assert choose_shop_command(game_state, {"choose", "proceed"}, want_to_shop=False) == "PROCEED"


def test_choose_shop_command_buys_first_affordable_relic() -> None:
    game_state = {
        "screen_type": "SHOP_SCREEN",
        "gold": 100,
        "screen_state": {
            "relics": [
                {"price": 90, "name": "Anchor"},
                {"price": 150, "name": "Bag of Prep"},
            ],
            "cards": [{"price": 50, "name": "Shrug It Off"}],
            "potions": [{"price": 40, "name": "Block Potion"}],
        },
    }

    assert choose_shop_command(game_state, {"leave", "state"}, want_to_shop=True) == "CHOOSE 0"


def test_choose_shop_command_buys_card_when_no_relic_is_affordable() -> None:
    game_state = {
        "screen_type": "SHOP_SCREEN",
        "gold": 60,
        "screen_state": {
            "relics": [{"price": 120, "name": "Anchor"}],
            "cards": [
                {"price": 70, "name": "Shrug It Off"},
                {"price": 55, "name": "Pommel Strike"},
            ],
            "potions": [{"price": 40, "name": "Block Potion"}],
        },
    }

    assert choose_shop_command(game_state, {"leave", "state"}, want_to_shop=True) == "CHOOSE 2"


def test_choose_shop_command_leaves_when_nothing_is_affordable() -> None:
    game_state = {
        "screen_type": "SHOP_SCREEN",
        "gold": 10,
        "screen_state": {
            "relics": [{"price": 120, "name": "Anchor"}],
            "cards": [{"price": 70, "name": "Shrug It Off"}],
            "potions": [{"price": 40, "name": "Block Potion"}],
        },
    }

    assert choose_shop_command(game_state, {"leave", "state"}, want_to_shop=True) == "LEAVE"


def test_choose_shop_command_leaves_immediately_when_disabled() -> None:
    game_state = {
        "screen_type": "SHOP_SCREEN",
        "gold": 60,
        "screen_state": {
            "relics": [{"price": 120, "name": "Anchor"}],
            "cards": [{"price": 55, "name": "Pommel Strike"}],
            "potions": [],
        },
    }

    assert choose_shop_command(game_state, {"leave", "state"}, want_to_shop=False) == "LEAVE"


def test_choose_play_command_uses_first_playable_targeted_card() -> None:
    combat_state = {
        "hand": [
            {"is_playable": True, "has_target": True},
            {"is_playable": True, "has_target": False},
        ],
        "monsters": [{"current_hp": 10, "is_gone": False}],
    }

    assert choose_play_command(combat_state) == "PLAY 1 0"


def test_choose_play_command_skips_targeted_card_without_targets() -> None:
    combat_state = {
        "hand": [
            {"is_playable": True, "has_target": True},
            {"is_playable": True, "has_target": False},
        ],
        "monsters": [],
    }

    assert choose_play_command(combat_state) == "PLAY 2"


def test_choose_combat_command_plays_card_when_waiting_on_user() -> None:
    game_state = {
        "room_phase": "COMBAT",
        "action_phase": "WAITING_ON_USER",
        "combat_state": {
            "hand": [{"is_playable": True, "has_target": False}],
            "monsters": [{"current_hp": 10, "is_gone": False}],
        },
    }

    assert choose_combat_command(game_state, {"play", "end"}) == "PLAY 1"


def test_choose_in_game_command_prefers_confirmation_before_other_actions() -> None:
    message = {
        "available_commands": ["proceed", "state"],
        "game_state": {"screen_type": "GRID"},
    }

    assert choose_in_game_command(message, last_command="CHOOSE 0", want_to_shop=False) == "PROCEED"


def test_choose_combat_command_ends_turn_when_no_play_exists() -> None:
    game_state = {
        "room_phase": "COMBAT",
        "action_phase": "WAITING_ON_USER",
        "combat_state": {
            "hand": [{"is_playable": False, "has_target": False}],
            "monsters": [{"current_hp": 10, "is_gone": False}],
        },
    }

    assert choose_combat_command(game_state, {"play", "end"}) == "END"


def test_append_log_writes_jsonl_entry(tmp_path) -> None:
    log_path = tmp_path / "communication_bridge_smoke.jsonl"

    append_log(log_path, '{"in_game": true}', {"in_game": True}, command="WAIT 30")

    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["command"] == "WAIT 30"
    assert entry["raw_line"] == '{"in_game": true}'
    assert entry["parsed_message"] == {"in_game": True}
    assert entry["summary"] == {
        "available_commands": None,
        "in_game": True,
        "ready_for_command": None,
    }
    assert "timestamp" in entry


def test_summarize_message_extracts_gameplay_fields() -> None:
    message = {
        "available_commands": ["play", "end", "state"],
        "in_game": True,
        "ready_for_command": True,
        "game_state": {
            "screen_type": "NONE",
            "screen_name": "NONE",
            "room_phase": "COMBAT",
            "action_phase": "WAITING_ON_USER",
            "act": 1,
            "floor": 1,
            "class": "IRONCLAD",
            "ascension_level": 20,
            "seed": 123,
            "combat_state": {
                "turn": 1,
                "hand": [{}, {}],
                "draw_pile": [{}, {}, {}],
                "discard_pile": [],
                "exhaust_pile": [],
                "player": {
                    "current_hp": 70,
                    "max_hp": 80,
                    "block": 4,
                    "energy": 3,
                },
                "monsters": [
                    {
                        "name": "Jaw Worm",
                        "id": "JawWorm",
                        "current_hp": 40,
                        "max_hp": 42,
                        "block": 0,
                        "intent": "ATTACK",
                        "move_id": 1,
                        "is_gone": False,
                    }
                ],
            },
        },
    }

    assert summarize_message(message) == {
        "available_commands": ["play", "end", "state"],
        "in_game": True,
        "ready_for_command": True,
        "screen_type": "NONE",
        "screen_name": "NONE",
        "room_phase": "COMBAT",
        "action_phase": "WAITING_ON_USER",
        "act": 1,
        "floor": 1,
        "class": "IRONCLAD",
        "ascension_level": 20,
        "seed": 123,
        "player": {
            "current_hp": 70,
            "max_hp": 80,
            "block": 4,
            "energy": 3,
        },
        "combat": {
            "turn": 1,
            "hand_size": 2,
            "draw_pile_size": 3,
            "discard_pile_size": 0,
            "exhaust_pile_size": 0,
            "monsters": [
                {
                    "index": 0,
                    "name": "Jaw Worm",
                    "id": "JawWorm",
                    "current_hp": 40,
                    "max_hp": 42,
                    "block": 0,
                    "intent": "ATTACK",
                    "move_id": 1,
                    "is_gone": False,
                }
            ],
        },
    }


def test_fingerprint_message_is_stable_for_equal_payloads() -> None:
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}

    assert fingerprint_message(left) == fingerprint_message(right)


def test_extract_run_seed_reads_seed_from_game_state() -> None:
    message = {"game_state": {"seed": 123456}}

    assert extract_run_seed(message) == 123456


def test_build_run_info_extracts_core_run_metadata() -> None:
    message = {
        "in_game": True,
        "available_commands": ["play", "end"],
        "game_state": {
            "seed": 123456,
            "class": "IRONCLAD",
            "ascension_level": 10,
            "act": 2,
            "floor": 17,
            "screen_type": "MAP",
            "screen_name": "MAP",
            "room_phase": "COMPLETE",
        },
    }

    info = build_run_info(message)

    assert info["seed"] == 123456
    assert info["class"] == "IRONCLAD"
    assert info["ascension_level"] == 10
    assert info["act"] == 2
    assert info["floor"] == 17
    assert info["screen_type"] == "MAP"
    assert info["available_commands"] == ["play", "end"]
    assert info["in_game"] is True
    assert "last_updated" in info


def test_persist_run_artifacts_writes_seed_scoped_files(tmp_path) -> None:
    message = {
        "in_game": True,
        "available_commands": ["wait", "state"],
        "ready_for_command": True,
        "game_state": {
            "seed": 999,
            "class": "IRONCLAD",
            "ascension_level": 0,
            "act": 1,
            "floor": 1,
            "screen_type": "NONE",
            "screen_name": "NONE",
            "room_phase": "COMBAT",
        },
    }

    persist_run_artifacts(tmp_path, message, command="WAIT 30", raw_line='{"seed":999}')

    run_dir = tmp_path / "seed_999"
    assert run_dir.exists()
    assert (run_dir / "observed_states.jsonl").exists()
    assert (run_dir / "bridge_commands.jsonl").exists()
    assert (run_dir / "latest_state.json").exists()
    assert (run_dir / "run_info.json").exists()

    latest_state = json.loads((run_dir / "latest_state.json").read_text(encoding="utf-8"))
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))

    assert latest_state["command"] == "WAIT 30"
    assert latest_state["parsed_message"]["game_state"]["seed"] == 999
    assert run_info["seed"] == 999


def test_first_living_monster_index_ignores_dead_or_gone_monsters() -> None:
    monsters = [
        {"current_hp": 0, "is_gone": False},
        {"current_hp": 10, "is_gone": True},
        {"current_hp": 15, "is_gone": False},
    ]

    assert first_living_monster_index(monsters) == 2


def test_first_affordable_index_returns_first_affordable_item() -> None:
    items = [
        {"price": 80},
        {"price": 40},
        {"price": 35},
    ]

    assert first_affordable_index(items, 50) == 1
