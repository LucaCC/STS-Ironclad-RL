"""Shared CLI helpers for the live rollout entrypoints."""

from __future__ import annotations

import importlib
from typing import Any

from sts_ironclad_rl.integration import BridgeConfig, BridgeTransport, LiveGameBridge
from sts_ironclad_rl.live import (
    BridgeObservationEncoder,
    CommunicationModActionContract,
    LiveEpisodeRunner,
    Policy,
    RandomLegalPolicy,
    ReplaySink,
    RunnerConfig,
    SimpleHeuristicPolicy,
)


def build_live_episode_runner(
    *,
    transport: BridgeTransport,
    host: str,
    port: int,
    max_steps: int,
    replay_sink: ReplaySink | None = None,
) -> LiveEpisodeRunner:
    """Build the default bridge-backed rollout runner used by CLI scripts."""

    bridge = LiveGameBridge(
        transport=transport,
        config=BridgeConfig(host=host, port=port),
    )
    return LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=BridgeObservationEncoder(),
        action_contract=CommunicationModActionContract(),
        replay_sink=replay_sink,
        config=RunnerConfig(max_steps=max_steps),
    )


def load_policy(policy_arg: str, *, seed: int | None) -> Policy:
    """Resolve one built-in or user-provided policy factory."""

    if policy_arg in {"simple_heuristic", "heuristic"}:
        return SimpleHeuristicPolicy()
    if policy_arg in {"random_legal", "random"}:
        return RandomLegalPolicy(seed=seed)

    factory = load_object(policy_arg)
    policy = factory()
    if not hasattr(policy, "select_action") or not hasattr(policy, "name"):
        raise TypeError("custom policy factory must return a live Policy-compatible object")
    return policy


def instantiate_transport(factory: Any) -> BridgeTransport:
    """Instantiate and validate one bridge transport factory."""

    transport = factory()
    if not isinstance(transport, BridgeTransport):
        raise TypeError("transport factory must return a BridgeTransport instance")
    return transport


def load_object(import_path: str) -> Any:
    """Resolve `module:attribute` into a Python object."""

    if ":" not in import_path:
        raise ValueError("import path must look like package.module:attribute")
    module_name, attribute_name = import_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)
