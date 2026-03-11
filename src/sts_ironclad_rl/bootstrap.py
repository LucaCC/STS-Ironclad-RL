"""Bootstrap primitives for the initial project scaffold."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectInfo:
    """Static metadata exposed by the initial package scaffold."""

    name: str
    supports_deterministic_seeds: bool


def get_project_info() -> ProjectInfo:
    """Return basic package metadata used by the smoke tests."""
    return ProjectInfo(
        name="sts-ironclad-rl",
        supports_deterministic_seeds=True,
    )
