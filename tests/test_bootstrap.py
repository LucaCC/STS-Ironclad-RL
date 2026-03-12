from sts_ironclad_rl import (
    __version__,
    get_project_info,
)
from sts_ironclad_rl.integration import (
    ActionCommand,
    BridgeConfig,
    BridgeEnvelope,
    BridgeMessageType,
    GameStateSnapshot,
    JsonlTrajectoryLogger,
    LiveGameBridge,
    LiveGameBridgeSession,
    TrajectoryEntry,
)


def test_package_metadata_smoke() -> None:
    info = get_project_info()

    assert __version__ == "0.1.0"
    assert info.name == "sts-ironclad-rl"
    assert info.supports_deterministic_seeds is True


def test_integration_exports_smoke() -> None:
    assert BridgeConfig is not None
    assert BridgeEnvelope is not None
    assert BridgeMessageType is not None
    assert GameStateSnapshot is not None
    assert ActionCommand is not None
    assert TrajectoryEntry is not None
    assert JsonlTrajectoryLogger is not None
    assert LiveGameBridge is not None
    assert LiveGameBridgeSession is not None
