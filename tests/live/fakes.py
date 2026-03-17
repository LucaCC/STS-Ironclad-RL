"""Low-overhead bridge and policy doubles for live tests."""

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
from sts_ironclad_rl.live import ActionDecision, EncodedObservation, ReplayEntry


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


@dataclass
class SequencedPolicy:
    decisions: tuple[ActionDecision, ...]
    name: str = "sequenced"
    _call_count: int = field(init=False, default=0, repr=False)

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        del observation
        if self._call_count >= len(self.decisions):
            msg = "policy was asked for more decisions than planned"
            raise AssertionError(msg)
        decision = self.decisions[self._call_count]
        self._call_count += 1
        return decision


@dataclass(frozen=True)
class InvalidPolicy:
    name: str = "invalid"

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        del observation
        return ActionDecision(action_id="illegal")


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


def build_bridge_with_session(
    snapshots: list[GameStateSnapshot] | None = None,
) -> tuple[LiveGameBridge, FakeTransport, str]:
    transport = FakeTransport()
    bridge = LiveGameBridge(transport=transport)
    session = bridge.connect()
    if snapshots is not None:
        transport.received.extend(
            [
                BridgeEnvelope.from_message(
                    BridgeMessageType.GAME_STATE,
                    GameStateSnapshot(
                        session_id=session.session_id,
                        screen_state=snapshot.screen_state,
                        available_actions=snapshot.available_actions,
                        in_combat=snapshot.in_combat,
                        floor=snapshot.floor,
                        act=snapshot.act,
                        raw_state=snapshot.raw_state,
                    ),
                )
                for snapshot in snapshots
            ]
        )
    return bridge, transport, session.session_id
