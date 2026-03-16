#!/usr/bin/env python3
"""Minimal CommunicationMod smoke-test and action-driver process.

This script is intentionally conservative. It:

- prints ``ready`` so CommunicationMod keeps the process alive
- logs every incoming line to a JSONL file
- replies with a low-risk command so the protocol continues

It does not implement the repo's full typed bridge. It is a host-side tool for
validating that CommunicationMod can launch a Python process, exchange
messages, and perform a few simple in-game actions.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_log_path() -> Path:
    """Return the default JSONL path for smoke-test protocol logs."""
    return Path(__file__).resolve().parents[1] / "logs" / "communication_bridge_smoke.jsonl"


def default_run_log_root() -> Path:
    """Return the root directory for per-run artifacts."""
    return Path(__file__).resolve().parents[1] / "logs" / "communication_runs"


def choose_command(
    message: dict[str, Any],
    *,
    autostart_command: str | None = "START IRONCLAD",
    has_started_run: bool = False,
    in_game_command: str = "WAIT 30",
    last_command: str | None = None,
    menu_command: str = "STATE",
    want_to_shop: bool = False,
) -> str:
    """Return a simple command based on the current CommunicationMod state."""
    if "error" in message:
        return "STATE"

    if not message.get("ready_for_command", False):
        return "STATE"

    available_commands = set(message.get("available_commands", []))
    if (
        not message.get("in_game", False)
        and not has_started_run
        and autostart_command
        and "start" in available_commands
    ):
        return autostart_command

    if message.get("in_game", False):
        scripted_command = choose_in_game_command(
            message,
            last_command=last_command,
            want_to_shop=want_to_shop,
        )
        if scripted_command is not None:
            return scripted_command
        return in_game_command

    return menu_command


def choose_in_game_command(
    message: dict[str, Any],
    *,
    last_command: str | None = None,
    want_to_shop: bool = False,
) -> str | None:
    """Return a simple in-game command for choices, combat, or navigation."""
    available_commands = set(message.get("available_commands", []))
    game_state = message.get("game_state")
    if not isinstance(game_state, dict):
        return None

    confirmation_command = choose_confirmation_command(
        game_state,
        available_commands,
        last_command=last_command,
    )
    if confirmation_command is not None:
        return confirmation_command

    shop_command = choose_shop_command(
        game_state,
        available_commands,
        want_to_shop=want_to_shop,
    )
    if shop_command is not None:
        return shop_command

    choice_command = choose_choice_command(game_state, available_commands)
    if choice_command is not None:
        return choice_command

    combat_command = choose_combat_command(game_state, available_commands)
    if combat_command is not None:
        return combat_command

    if "proceed" in available_commands:
        return "PROCEED"
    if "return" in available_commands:
        return "RETURN"
    return None


def choose_confirmation_command(
    game_state: dict[str, Any],
    available_commands: set[str],
    *,
    last_command: str | None,
) -> str | None:
    """Confirm a previously selected option when the game exposes proceed."""
    if "proceed" not in available_commands:
        return None
    if not last_command or not last_command.startswith("CHOOSE"):
        return None

    # Avoid auto-proceeding past map path selection before we intentionally pick
    # a node or shop-room entry where CHOOSE is still the primary action.
    if game_state.get("screen_type") in {"MAP", "SHOP_ROOM"}:
        return None

    return "PROCEED"


def choose_shop_command(
    game_state: dict[str, Any],
    available_commands: set[str],
    *,
    want_to_shop: bool,
) -> str | None:
    """Return a simple shop action, or leave if nothing affordable is available."""
    if game_state.get("screen_type") == "SHOP_ROOM":
        if want_to_shop:
            return choose_choice_command(game_state, available_commands)
        if "proceed" in available_commands:
            return "PROCEED"
        return None

    if game_state.get("screen_type") != "SHOP_SCREEN":
        return None

    if not want_to_shop:
        return "LEAVE" if "leave" in available_commands else None

    screen_state = game_state.get("screen_state")
    if not isinstance(screen_state, dict):
        return "LEAVE" if "leave" in available_commands else None

    gold = game_state.get("gold")
    if not isinstance(gold, int):
        gold = 0

    affordable_relic = first_affordable_index(screen_state.get("relics"), gold)
    if affordable_relic is not None:
        return f"CHOOSE {affordable_relic}"

    cards = screen_state.get("cards")
    affordable_card = first_affordable_index(cards, gold)
    if affordable_card is not None:
        relic_count = (
            len(screen_state.get("relics", []))
            if isinstance(screen_state.get("relics"), list)
            else 0
        )
        return f"CHOOSE {relic_count + affordable_card}"

    affordable_potion = first_affordable_index(screen_state.get("potions"), gold)
    if affordable_potion is not None:
        relic_count = (
            len(screen_state.get("relics", []))
            if isinstance(screen_state.get("relics"), list)
            else 0
        )
        card_count = len(cards) if isinstance(cards, list) else 0
        return f"CHOOSE {relic_count + card_count + affordable_potion}"

    if "leave" in available_commands:
        return "LEAVE"
    return None


def first_affordable_index(items: Any, gold: int) -> int | None:
    """Return the first affordable item index from a shop item list."""
    if not isinstance(items, list):
        return None

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        price = item.get("price")
        if isinstance(price, int) and price <= gold:
            return index
    return None


def choose_choice_command(game_state: dict[str, Any], available_commands: set[str]) -> str | None:
    """Return the first available choice command when the screen exposes choices."""
    if "choose" not in available_commands:
        return None

    choice_list = game_state.get("choice_list")
    if not isinstance(choice_list, list) or not choice_list:
        return None

    first_choice = choice_list[0]
    if isinstance(first_choice, str) and first_choice and " " not in first_choice:
        return f"CHOOSE {first_choice}"

    # Choice-name parsing with spaces is unclear from the mod README, so fall
    # back to the first index for broad compatibility.
    return "CHOOSE 0"


def choose_combat_command(game_state: dict[str, Any], available_commands: set[str]) -> str | None:
    """Return a basic combat action if the game is waiting on user input."""
    combat_state = game_state.get("combat_state")
    if not isinstance(combat_state, dict):
        return None

    if game_state.get("room_phase") != "COMBAT":
        return None
    if game_state.get("action_phase") != "WAITING_ON_USER":
        return None

    if "play" in available_commands:
        play_command = choose_play_command(combat_state)
        if play_command is not None:
            return play_command

    if "potion" in available_commands:
        potion_command = choose_potion_command(game_state)
        if potion_command is not None:
            return potion_command

    if "end" in available_commands:
        return "END"

    return None


def choose_play_command(combat_state: dict[str, Any]) -> str | None:
    """Return a basic card-play command using the first playable card."""
    hand = combat_state.get("hand")
    if not isinstance(hand, list):
        return None

    target_index = first_living_monster_index(combat_state.get("monsters"))
    for card_index, card in enumerate(hand, start=1):
        if not isinstance(card, dict):
            continue
        if not card.get("is_playable", False):
            continue

        if card.get("has_target", False):
            if target_index is None:
                continue
            return f"PLAY {card_index} {target_index}"
        return f"PLAY {card_index}"

    return None


def choose_potion_command(game_state: dict[str, Any]) -> str | None:
    """Return a potion-use command if a usable potion is available."""
    potions = game_state.get("potions")
    if not isinstance(potions, list):
        return None

    target_index = first_living_monster_index(game_state.get("combat_state", {}).get("monsters"))
    for slot_index, potion in enumerate(potions):
        if not isinstance(potion, dict):
            continue
        if not potion.get("can_use", False):
            continue

        if potion.get("requires_target", False):
            if target_index is None:
                continue
            return f"POTION USE {slot_index} {target_index}"
        return f"POTION USE {slot_index}"

    return None


def first_living_monster_index(monsters: Any) -> int | None:
    """Return the first monster index that still appears to be alive."""
    if not isinstance(monsters, list):
        return None

    for index, monster in enumerate(monsters):
        if not isinstance(monster, dict):
            continue
        if monster.get("is_gone", False):
            continue
        if monster.get("current_hp", 0) is None:
            continue
        if monster.get("current_hp", 0) <= 0:
            continue
        return index
    return None


def summarize_message(parsed_message: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the fields most useful for building the real bridge."""
    if parsed_message is None:
        return None

    summary: dict[str, Any] = {
        "available_commands": parsed_message.get("available_commands"),
        "in_game": parsed_message.get("in_game"),
        "ready_for_command": parsed_message.get("ready_for_command"),
    }
    if "error" in parsed_message:
        summary["error"] = parsed_message["error"]
        return summary

    game_state = parsed_message.get("game_state")
    if not isinstance(game_state, dict):
        return summary

    summary.update(
        {
            "screen_type": game_state.get("screen_type"),
            "screen_name": game_state.get("screen_name"),
            "room_phase": game_state.get("room_phase"),
            "action_phase": game_state.get("action_phase"),
            "act": game_state.get("act"),
            "floor": game_state.get("floor"),
            "class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "seed": game_state.get("seed"),
        }
    )

    player = game_state.get("combat_state", {}).get("player")
    if isinstance(player, dict):
        summary["player"] = {
            "current_hp": player.get("current_hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "energy": player.get("energy"),
        }

    combat_state = game_state.get("combat_state")
    if isinstance(combat_state, dict):
        summary["combat"] = {
            "turn": combat_state.get("turn"),
            "hand_size": len(combat_state.get("hand", [])),
            "draw_pile_size": len(combat_state.get("draw_pile", [])),
            "discard_pile_size": len(combat_state.get("discard_pile", [])),
            "exhaust_pile_size": len(combat_state.get("exhaust_pile", [])),
            "monsters": summarize_monsters(combat_state.get("monsters")),
        }

    return summary


def summarize_monsters(monsters: Any) -> list[dict[str, Any]]:
    """Extract the monster details most relevant to action translation."""
    if not isinstance(monsters, list):
        return []

    summary: list[dict[str, Any]] = []
    for index, monster in enumerate(monsters):
        if not isinstance(monster, dict):
            continue
        summary.append(
            {
                "index": index,
                "name": monster.get("name"),
                "id": monster.get("id"),
                "current_hp": monster.get("current_hp"),
                "max_hp": monster.get("max_hp"),
                "block": monster.get("block"),
                "intent": monster.get("intent"),
                "move_id": monster.get("move_id"),
                "is_gone": monster.get("is_gone"),
            }
        )
    return summary


def fingerprint_message(parsed_message: dict[str, Any] | None) -> str:
    """Return a stable fingerprint for duplicate suppression."""
    return json.dumps(parsed_message, sort_keys=True)


def extract_run_seed(parsed_message: dict[str, Any] | None) -> int | None:
    """Return the run seed from a CommunicationMod payload if available."""
    if not isinstance(parsed_message, dict):
        return None

    game_state = parsed_message.get("game_state")
    if not isinstance(game_state, dict):
        return None

    seed = game_state.get("seed")
    return seed if isinstance(seed, int) else None


def build_run_info(parsed_message: dict[str, Any]) -> dict[str, Any]:
    """Extract stable run metadata for a per-seed info file."""
    game_state = parsed_message.get("game_state", {})
    info = {
        "seed": game_state.get("seed"),
        "class": game_state.get("class"),
        "ascension_level": game_state.get("ascension_level"),
        "act": game_state.get("act"),
        "floor": game_state.get("floor"),
        "screen_type": game_state.get("screen_type"),
        "screen_name": game_state.get("screen_name"),
        "room_phase": game_state.get("room_phase"),
        "in_game": parsed_message.get("in_game"),
        "available_commands": parsed_message.get("available_commands"),
        "last_updated": datetime.now(tz=timezone.utc).isoformat(),
    }
    return info


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON payload as a single JSONL line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def persist_run_artifacts(
    run_log_root: Path,
    parsed_message: dict[str, Any] | None,
    *,
    command: str,
    raw_line: str,
) -> None:
    """Persist per-run state, metadata, and command history when a seed is known."""
    seed = extract_run_seed(parsed_message)
    if seed is None or parsed_message is None:
        return

    run_dir = run_log_root / f"seed_{seed}"
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    summary = summarize_message(parsed_message)

    append_jsonl(
        run_dir / "observed_states.jsonl",
        {
            "timestamp": timestamp,
            "summary": summary,
        },
    )
    append_jsonl(
        run_dir / "bridge_commands.jsonl",
        {
            "timestamp": timestamp,
            "command": command,
            "available_commands": parsed_message.get("available_commands"),
        },
    )
    write_json(
        run_dir / "latest_state.json",
        {
            "timestamp": timestamp,
            "command": command,
            "raw_line": raw_line,
            "parsed_message": parsed_message,
            "summary": summary,
        },
    )
    write_json(run_dir / "run_info.json", build_run_info(parsed_message))


def append_log(
    log_path: Path,
    raw_line: str,
    parsed_message: dict[str, Any] | None,
    *,
    command: str,
) -> None:
    """Append one raw CommunicationMod line to the local smoke-test log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "command": command,
        "raw_line": raw_line,
        "parsed_message": parsed_message,
        "summary": summarize_message(parsed_message),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def main() -> int:
    """Run the minimal CommunicationMod bridge loop."""
    log_path = Path(os.environ.get("STS_COMM_LOG_PATH", default_log_path()))
    run_log_root = Path(os.environ.get("STS_COMM_RUN_LOG_ROOT", default_run_log_root()))
    autostart_command = os.environ.get("STS_COMM_AUTOSTART", "START IRONCLAD")
    in_game_command = os.environ.get("STS_COMM_IN_GAME_COMMAND", "WAIT 30")
    menu_command = os.environ.get("STS_COMM_MENU_COMMAND", "STATE")
    want_to_shop = os.environ.get("STS_COMM_WANT_TO_SHOP", "").lower() in {
        "1",
        "true",
        "yes",
    }
    log_duplicates = os.environ.get("STS_COMM_LOG_DUPLICATES", "").lower() in {
        "1",
        "true",
        "yes",
    }
    has_started_run = False
    last_command: str | None = None
    last_fingerprint: str | None = None

    print("ready", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        parsed_message: dict[str, Any] | None = None
        command = "STATE"
        try:
            parsed_message = json.loads(line)
            if isinstance(parsed_message, dict):
                command = choose_command(
                    parsed_message,
                    autostart_command=autostart_command,
                    has_started_run=has_started_run,
                    in_game_command=in_game_command,
                    last_command=last_command,
                    menu_command=menu_command,
                    want_to_shop=want_to_shop,
                )
                if command.upper().startswith("START"):
                    has_started_run = True
            else:
                parsed_message = {"unexpected_payload": parsed_message}
        except json.JSONDecodeError as exc:
            parsed_message = {"decode_error": str(exc)}

        fingerprint = fingerprint_message(parsed_message)
        if log_duplicates or fingerprint != last_fingerprint:
            append_log(log_path, line, parsed_message, command=command)
            persist_run_artifacts(
                run_log_root,
                parsed_message,
                command=command,
                raw_line=line,
            )
            last_fingerprint = fingerprint

        print(command, flush=True)
        last_command = command

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
