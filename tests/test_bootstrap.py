from sts_ironclad_rl import (
    __version__,
    agents,
    evaluation,
    get_project_info,
    training,
    utils,
)


def test_package_metadata_smoke() -> None:
    info = get_project_info()

    assert __version__ == "0.1.0"
    assert info.name == "sts-ironclad-rl"
    assert info.supports_deterministic_seeds is True


def test_package_layout_smoke() -> None:
    assert agents.__name__ == "sts_ironclad_rl.agents"
    assert training.__name__ == "sts_ironclad_rl.training"
    assert evaluation.__name__ == "sts_ironclad_rl.evaluation"
    assert utils.__name__ == "sts_ironclad_rl.utils"
