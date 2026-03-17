from __future__ import annotations

from dataclasses import dataclass

import pytest

from sts_ironclad_rl.live import (
    ActionDecision,
    EncodedObservation,
    LiveEpisodeRunner,
    RawStateObservationEncoder,
    RunnerConfig,
    SnapshotActionContract,
)
from tests.live.factories import make_snapshot
from tests.live.fakes import ReplayCollector, SequencedPolicy, build_bridge_with_session


def test_snapshot_action_contract_preserves_bridge_namespace_and_arguments() -> None:
    snapshot = make_snapshot(available_actions=("play", "end"))
    decision = ActionDecision(action_id="play", arguments={"card_index": 2})
    contract = SnapshotActionContract()

    assert contract.legal_action_ids(snapshot) == ("play", "end")
    assert contract.to_validated_command(snapshot, decision).arguments == {"card_index": 2}


def test_snapshot_action_contract_rejects_illegal_actions() -> None:
    snapshot = make_snapshot(available_actions=("end",))

    with pytest.raises(ValueError, match="illegal action"):
        SnapshotActionContract().to_validated_command(
            snapshot,
            ActionDecision(action_id="play"),
        )


@dataclass(frozen=True)
class EmptyLegalActionEncoder:
    def encode(self, snapshot) -> EncodedObservation:
        return EncodedObservation(
            snapshot=snapshot,
            legal_action_ids=(),
            features={"turn": 1},
            metadata={},
        )


def test_rollout_runner_falls_back_to_action_contract_legal_actions() -> None:
    bridge, transport, session_id = build_bridge_with_session(
        snapshots=[
            make_snapshot(
                session_id="placeholder",
                available_actions=("end",),
                raw_state={"turn": 1},
            ),
            make_snapshot(
                session_id="placeholder",
                screen_state="MAP",
                available_actions=("proceed",),
                in_combat=False,
                floor=4,
                raw_state={"victory": True},
            ),
        ]
    )
    replay_sink = ReplayCollector()
    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=EmptyLegalActionEncoder(),
        action_contract=SnapshotActionContract(),
        replay_sink=replay_sink,
        config=RunnerConfig(max_steps=3),
    )

    result = runner.run_episode(
        policy=SequencedPolicy(decisions=(ActionDecision(action_id="end"),))
    )

    assert result.terminal is True
    assert result.outcome == "victory"
    assert result.session_id == session_id
    assert replay_sink.entries[0].action == ActionDecision(action_id="end")
    assert transport.sent[-2].payload["command"] == "end"


def test_raw_state_observation_encoder_keeps_snapshot_namespace_intact() -> None:
    snapshot = make_snapshot(
        available_actions=("play",),
        raw_state={"turn": 2, "reward": 1.5},
    )
    observation = RawStateObservationEncoder().encode(snapshot)

    assert observation.snapshot is snapshot
    assert observation.legal_action_ids == ("play",)
    assert observation.features == {"turn": 2, "reward": 1.5}
