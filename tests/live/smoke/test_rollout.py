from __future__ import annotations

from sts_ironclad_rl.integration import BridgeMessageType
from sts_ironclad_rl.live import (
    ActionDecision,
    BridgeObservationEncoder,
    CommunicationModActionContract,
    LiveEpisodeRunner,
    RawStateObservationEncoder,
    RunnerConfig,
    SimpleHeuristicPolicy,
    SnapshotActionContract,
)
from tests.live.factories import make_card, make_combat_snapshot, make_enemy, make_event_snapshot
from tests.live.fakes import (
    InvalidPolicy,
    ReplayCollector,
    SequencedPolicy,
    WrongSessionEncoder,
    build_bridge_with_session,
)


def test_live_episode_runner_completes_episode_and_records_metrics() -> None:
    bridge, transport, session_id = build_bridge_with_session(
        snapshots=[
            make_combat_snapshot(
                session_id="placeholder",
                available_actions=("play_0", "end_turn"),
                turn=1,
                extra_raw_state={"turn": 1, "reward": 1.5},
            ),
            make_combat_snapshot(
                session_id="placeholder",
                available_actions=("end_turn",),
                turn=2,
                extra_raw_state={"turn": 2, "reward": 0.5},
            ),
            make_event_snapshot(
                session_id="placeholder",
                available_actions=("proceed",),
                screen_state="MAP",
                extra_raw_state={"victory": True, "score": 33, "reward": 2.0},
            ),
        ]
    )
    replay_sink = ReplayCollector()
    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=RawStateObservationEncoder(),
        action_contract=SnapshotActionContract(),
        replay_sink=replay_sink,
    )

    result = runner.run_episode(
        policy=SequencedPolicy(
            decisions=(
                ActionDecision(action_id="play_0"),
                ActionDecision(action_id="end_turn"),
            )
        )
    )

    assert result.outcome == "victory"
    assert result.terminal is True
    assert result.step_count == 2
    assert len(result.entries) == 3
    assert replay_sink.entries == list(result.entries)
    assert result.session_id == session_id
    assert result.total_reward == 4.0
    assert result.metadata["final_score"] == 33
    assert result.metadata["final_floor"] == 3
    assert transport.closed is True
    assert [envelope.message_type for envelope in transport.sent] == [
        BridgeMessageType.SESSION_HELLO,
        BridgeMessageType.REQUEST_STATE,
        BridgeMessageType.ACTION_COMMAND,
        BridgeMessageType.REQUEST_STATE,
        BridgeMessageType.ACTION_COMMAND,
        BridgeMessageType.REQUEST_STATE,
    ]


def test_live_episode_runner_reports_bridge_disconnect() -> None:
    bridge, transport, _ = build_bridge_with_session(snapshots=[])
    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=RawStateObservationEncoder(),
        action_contract=SnapshotActionContract(),
        config=RunnerConfig(max_empty_polls_per_step=2),
    )

    result = runner.run_episode(
        policy=SequencedPolicy(decisions=(ActionDecision(action_id="end_turn"),))
    )

    assert result.outcome == "interrupted"
    assert result.failure is not None
    assert result.failure.kind == "bridge_disconnect"
    assert transport.closed is True


def test_live_episode_runner_reports_invalid_policy_output() -> None:
    bridge, _, _ = build_bridge_with_session(
        snapshots=[make_combat_snapshot(session_id="placeholder", available_actions=("end_turn",))]
    )
    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=RawStateObservationEncoder(),
        action_contract=SnapshotActionContract(),
    )

    result = runner.run_episode(policy=InvalidPolicy())

    assert result.outcome == "interrupted"
    assert result.failure is not None
    assert result.failure.kind == "invalid_policy_output"


def test_live_episode_runner_reports_malformed_state() -> None:
    bridge, _, _ = build_bridge_with_session(
        snapshots=[make_combat_snapshot(session_id="placeholder", available_actions=("end_turn",))]
    )
    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=WrongSessionEncoder(),
        action_contract=SnapshotActionContract(),
    )

    result = runner.run_episode(
        policy=SequencedPolicy(decisions=(ActionDecision(action_id="end_turn"),))
    )

    assert result.outcome == "interrupted"
    assert result.failure is not None
    assert result.failure.kind == "malformed_state"


def test_live_episode_runner_smoke_exercises_bridge_observation_action_and_policy() -> None:
    bridge, transport, _ = build_bridge_with_session(
        snapshots=[
            make_combat_snapshot(
                session_id="placeholder",
                hand=(
                    make_card(name="Strike", has_target=True),
                    make_card(name="Defend"),
                ),
                monsters=(
                    make_enemy(name="Jaw Worm", current_hp=19, intent_base_damage=12),
                    make_enemy(name="Cultist", current_hp=7, max_hp=48),
                ),
                player={"current_hp": 62, "max_hp": 80, "block": 0, "energy": 2},
            ),
            make_event_snapshot(
                session_id="placeholder",
                available_actions=("proceed",),
                screen_state="MAP",
                extra_raw_state={"victory": True, "score": 21},
            ),
        ]
    )
    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=BridgeObservationEncoder(),
        action_contract=CommunicationModActionContract(),
    )

    result = runner.run_episode(policy=SimpleHeuristicPolicy())

    assert result.terminal is True
    assert result.outcome == "victory"
    assert result.step_count == 1
    assert transport.sent[-2].payload == {
        "session_id": result.session_id,
        "command": "play",
        "arguments": {"card_index": 1, "target_index": 1},
    }
