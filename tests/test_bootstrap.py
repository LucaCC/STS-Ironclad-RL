from sts_ironclad_rl import __version__, get_project_info


def test_package_metadata_smoke() -> None:
    info = get_project_info()

    assert __version__ == "0.1.0"
    assert info.name == "sts-ironclad-rl"
    assert info.supports_deterministic_seeds is True
