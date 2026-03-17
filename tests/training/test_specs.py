from __future__ import annotations

import json

import pytest

from sts_ironclad_rl.training import ExperimentSpec, load_experiment_spec


def test_experiment_spec_round_trips_and_builds_evaluation_case(tmp_path) -> None:
    spec = ExperimentSpec.from_dict(
        {
            "experiment_name": "Random Legal Smoke",
            "policy_name": "random_legal",
            "episode_count": 4,
            "evaluation_case_name": "smoke",
            "max_steps": 12,
            "seed": 7,
            "tags": ["wave3", "collection"],
            "metadata": {"bridge": "fake"},
        }
    )

    config_path = tmp_path / "spec.json"
    config_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    loaded = load_experiment_spec(config_path)

    assert loaded == spec
    assert loaded.fingerprint() == spec.fingerprint()
    assert loaded.to_evaluation_case().name == "smoke"
    assert loaded.to_evaluation_case().metadata["experiment_name"] == "Random Legal Smoke"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"experiment_name": "", "policy_name": "random_legal", "episode_count": 1},
            "experiment_name",
        ),
        (
            {"experiment_name": "smoke", "policy_name": "", "episode_count": 1},
            "policy_name",
        ),
        (
            {"experiment_name": "smoke", "policy_name": "random_legal", "episode_count": 0},
            "episode_count",
        ),
        (
            {
                "experiment_name": "smoke",
                "policy_name": "random_legal",
                "episode_count": 1,
                "max_steps": 0,
            },
            "max_steps",
        ),
        (
            {
                "experiment_name": "smoke",
                "policy_name": "random_legal",
                "episode_count": 1,
                "metadata": {"bad": object()},
            },
            "metadata",
        ),
    ],
)
def test_experiment_spec_rejects_invalid_payloads(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentSpec.from_dict(payload)
