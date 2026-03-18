from __future__ import annotations

from sts_ironclad_rl.live import load_live_policy


def test_load_policy_accepts_canonical_builtin_names() -> None:
    assert load_live_policy("simple_heuristic", seed=None).name == "simple_heuristic"
    assert load_live_policy("random_legal", seed=7).name == "random_legal"


def test_load_policy_keeps_short_aliases_for_cli_compatibility() -> None:
    assert load_live_policy("heuristic", seed=None).name == "simple_heuristic"
    assert load_live_policy("random", seed=7).name == "random_legal"
