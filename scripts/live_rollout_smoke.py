# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
    EvaluationCase,
    LiveEpisodeRunner,
    PolicyContext,
    RawStateObservationEncoder,
    RunnerConfig,
    SnapshotActionContract,
)


@dataclass
class ScriptedTransport(BridgeTransport):
    snapshots: list[GameStateSnapshot]
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
        if not self.snapshots:
            return None
        snapshot = self.snapshots.pop(0)
        return BridgeEnvelope.from_message(BridgeMessageType.GAME_STATE, snapshot)


@dataclass(frozen=True)
class FirstLegalActionPolicy:
    name: str = "first_legal_action"

    def act(
        self,
        observation,
        legal_actions: tuple[str, ...],
        context: PolicyContext,
    ) -> ActionDecision:
        del observation
        del context
        return ActionDecision(action_id=legal_actions[0])


def build_scripted_snapshots(session_id: str) -> list[GameStateSnapshot]:
    return [
        GameStateSnapshot(
            session_id=session_id,
            screen_state="COMBAT",
            available_actions=("play_0", "end_turn"),
            in_combat=True,
            floor=1,
            act=1,
            raw_state={"turn": 1, "player_hp": 80},
        ),
        GameStateSnapshot(
            session_id=session_id,
            screen_state="COMBAT",
            available_actions=("end_turn",),
            in_combat=True,
            floor=1,
            act=1,
            raw_state={"turn": 2, "player_hp": 80},
        ),
        GameStateSnapshot(
            session_id=session_id,
            screen_state="MAP",
            available_actions=(),
            in_combat=False,
            floor=1,
            act=1,
            raw_state={"victory": True},
        ),
    ]


def main() -> None:
    transport = ScriptedTransport(snapshots=[])
    bridge = LiveGameBridge(transport=transport)
    session = bridge.connect()
    transport.snapshots.extend(build_scripted_snapshots(session.session_id))

    runner = LiveEpisodeRunner(
        bridge=bridge,
        observation_encoder=RawStateObservationEncoder(),
        action_contract=SnapshotActionContract(),
        config=RunnerConfig(max_steps=5),
    )
    result = runner.run_episode(
        policy=FirstLegalActionPolicy(),
        evaluation_case=EvaluationCase(name="scripted-smoke", max_steps=5),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
