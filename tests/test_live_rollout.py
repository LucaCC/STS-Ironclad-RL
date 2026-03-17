from __future__ import annotations

from dataclasses import dataclass, field

from sts_ironclad_rl.integration import (
    BridgeConfig,
    BridgeEnvelope,
    BridgeMessageType,
    GameStateSnapshot,
    LiveGameBridge,
)
from sts_ironclad_rl.integration.bridge import BridgeTransport
from sts_ironclad_rl.live import (
    ActionDecision,
    EncodedObservation,
    EvaluationCase,
    LiveEpisodeRunner,
    PolicyContext,
    RawStateObservationEncoder,
    ReplayEntry,
    RunnerConfig,
    SnapshotActionContract,
)


@dataclass
class FakeTransport(BridgeTransport):
    received: list[BridgeEnvelope] = field(default_factory=list)
    sent: list[BridgeEnvelope] = field(default_factory=list)
    opened_with: BridgeConfig | None = None
    closed: bool = False

    def open(self, config: BridgeConfig) -> None:
        self.opened_with = config

    def close(self) -> None:
        self.closed = True

    def send(self, envelope: BridgeEnvelope) -> None:
        self.sent.append(envelope)

    def receive(self) -> BridgeEnvelope | None:
        if not self.received:
            return None
        return self.received.pop(0)


@dataclass
class ReplayCollector:
    entries: list[ReplayEntry] = field(default_factory=list)

    def log(self, entry: ReplayEntry) -> None:
        self.entries.append(entry)


@dataclass(frozen=True)
class PlannedPolicy:
    decisions: tuple[ActionDecision, ...]
    name: str = "planned"

    def act(
        self,
        observation: EncodedObservation,
        legal_actions: tuple[str, ...],
        context: PolicyContext,
    ) -> ActionDecision:
        del observation
        del legal_actions
        return self.decisions[context.step_index]


@dataclass(frozen=True)
class InvalidPolicy:
    name: str = "invalid"

    def act(
        self,
        observation: EncodedObservation,
        legal_actions: tuple[str, ...],
        context: PolicyContext,
    ) -> str:
        del observation
        del legal_actions
        del context
        return "end_turn"


@dataclass(frozen=True)
class WrongSessionEncoder:
    def encode(self, snapshot: GameStateSnapshot) -> EncodedObservation:
        return EncodedObservation(
            snapshot=GameStateSnapshot(
                session_id="wrong-session",
                screen_state=snapshot.screen_state,
                available_actions=snapshot.available_actions,
                in_combat=snapshot.in_combat,
                floor=snapshot.floor,
                act=snapshot.act,
                raw_state=snapshot.raw_state,
            ),
            legal_action_ids=snapshot.available_actions,
            features={},
            metadata={},
        )


def _game_state(snapshot: GameStateSnapshot) -> BridgeEnvelope:
    return BridgeEnvelope.from_message(BridgeMessageType.GAME_STATE, snapshot)


def _build_bridge_with_session(
    snapshots: list[GameStateSnapshot] | None = None,
) -> tuple[LiveGameBridge, FakeTransport, str]:
    transport = FakeTransport()
    bridge = LiveGameBridge(transport=transport)
    session = bridge.connect()
    if snapshots is not None:
        transport.received.extend(
            [
                _game_state(
                    GameStateSnapshot(
                        session_id=session.session_id,
                        screen_state=snapshot.screen_state,
                        available_actions=snapshot.available_actions,
                        in_combat=snapshot.in_combat,
                        floor=snapshot.floor,
                        act=snapshot.act,
                        raw_state=snapshot.raw_state,
                    )
                )
                for snapshot in snapshots
            ]
        )
    return bridge, transport, session.session_id


def test_live_episode_runner_completes_episode_and_logs_trace() -> None:
    bridge, transport, session_id = _build_bridge_with_session(
        snapshots=[
            GameStateSnapshot(
                session_id="placeholder",
                screen_state="COMBAT",
                available_actions=("play_0", "end_turn"),
                in_combat=True,
                floor=1,
                act=1,
                raw_state={"turn": 1},
            ),
            GameStateSnapshot(
                session_id="placeholder",
                screen_state="COMBAT",
                available_actions=("end_turn",),
                in_combat=True,
                floor=1,
                act=1,
                raw_state={"turn": 2},
            ),
            GameStateSnapshot(
                session_id="placeholder",
                screen_state="MAP",
                available_actions=(),
                in_combat=False,
                floor=1,
                act=1,
                raw_state={"victory": True},
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
        policy=PlannedPolicy(
            decisions=(
                ActionDecision(action_id="play_0", metadata={"source": "test"}),
                ActionDecision(action_id="end_turn"),
            )
        ),
        evaluation_case=EvaluationCase(name="smoke", max_steps=10),
    )

    assert result.outcome == "victory"
    assert result.terminal is True
    assert result.step_count == 2
    assert len(result.entries) == 3
    assert len(result.action_trace) == 2
    assert replay_sink.entries == list(result.entries)
    assert result.action_trace[0].action_id == "play_0"
    assert result.action_trace[0].metadata == {"source": "test"}
    assert result.session_id == session_id
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
    bridge, transport, _ = _build_bridge_with_session(snapshots=[])
    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=RawStateObservationEncoder(),
        action_contract=SnapshotActionContract(),
        config=RunnerConfig(max_empty_polls_per_step=2),
    )

    result = runner.run_episode(policy=PlannedPolicy(decisions=(ActionDecision(action_id="end"),)))

    assert result.outcome == "failure"
    assert result.failure is not None
    assert result.failure.kind == "bridge_disconnect"
    assert transport.closed is True


def test_live_episode_runner_reports_invalid_policy_output() -> None:
    bridge, _, _ = _build_bridge_with_session(
        snapshots=[
            GameStateSnapshot(
                session_id="placeholder",
                screen_state="COMBAT",
                available_actions=("end_turn",),
                in_combat=True,
                raw_state={},
            )
        ]
    )

    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=RawStateObservationEncoder(),
        action_contract=SnapshotActionContract(),
    )

    result = runner.run_episode(policy=InvalidPolicy())

    assert result.outcome == "failure"
    assert result.failure is not None
    assert result.failure.kind == "invalid_policy_output"


def test_live_episode_runner_reports_malformed_state() -> None:
    bridge, _, _ = _build_bridge_with_session(
        snapshots=[
            GameStateSnapshot(
                session_id="placeholder",
                screen_state="COMBAT",
                available_actions=("end_turn",),
                in_combat=True,
                raw_state={},
            )
        ]
    )

    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=WrongSessionEncoder(),
        action_contract=SnapshotActionContract(),
    )

    result = runner.run_episode(
        policy=PlannedPolicy(decisions=(ActionDecision(action_id="end_turn"),))
    )

    assert result.outcome == "failure"
    assert result.failure is not None
    assert result.failure.kind == "malformed_state"


def test_live_episode_runner_stops_on_repeated_no_ops() -> None:
    bridge, _, _ = _build_bridge_with_session(
        snapshots=[
            GameStateSnapshot(
                session_id="placeholder",
                screen_state="COMBAT",
                available_actions=("wait",),
                in_combat=True,
                raw_state={"turn": 1},
            ),
            GameStateSnapshot(
                session_id="placeholder",
                screen_state="COMBAT",
                available_actions=("wait",),
                in_combat=True,
                raw_state={"turn": 2},
            ),
            GameStateSnapshot(
                session_id="placeholder",
                screen_state="COMBAT",
                available_actions=("wait",),
                in_combat=True,
                raw_state={"turn": 3},
            ),
        ]
    )

    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=RawStateObservationEncoder(),
        action_contract=SnapshotActionContract(),
        config=RunnerConfig(max_repeated_no_ops=3, max_steps=5),
    )

    result = runner.run_episode(
        policy=PlannedPolicy(
            decisions=(
                ActionDecision(action_id="wait"),
                ActionDecision(action_id="wait"),
                ActionDecision(action_id="wait"),
            )
        )
    )

    assert result.outcome == "failure"
    assert result.failure is not None
    assert result.failure.kind == "no_op_guard"


def test_live_episode_runner_stops_on_repeated_state_fingerprint() -> None:
    repeated_snapshot = GameStateSnapshot(
        session_id="placeholder",
        screen_state="COMBAT",
        available_actions=("end_turn",),
        in_combat=True,
        raw_state={"turn": 1},
    )
    bridge, _, _ = _build_bridge_with_session(
        snapshots=[repeated_snapshot, repeated_snapshot, repeated_snapshot]
    )

    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=RawStateObservationEncoder(),
        action_contract=SnapshotActionContract(),
        config=RunnerConfig(max_repeated_state_fingerprints=2, max_steps=5),
    )

    result = runner.run_episode(
        policy=PlannedPolicy(
            decisions=(
                ActionDecision(action_id="end_turn"),
                ActionDecision(action_id="end_turn"),
            )
        )
    )

    assert result.outcome == "failure"
    assert result.failure is not None
    assert result.failure.kind == "desync_guard"
