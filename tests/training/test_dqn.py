from __future__ import annotations

import random

import pytest
import torch

from sts_ironclad_rl.training import (
    DEFAULT_ACTION_SIZE,
    DEFAULT_OBSERVATION_SIZE,
    MaskedDQN,
    ReplayBuffer,
    epsilon_greedy_action,
    legal_argmax,
    load_checkpoint,
    sanitize_legal_action_mask,
    save_checkpoint,
)


def make_transition(index: int) -> dict[str, object]:
    state = tuple(float(index + offset) for offset in range(DEFAULT_OBSERVATION_SIZE))
    next_state = tuple(value + 0.5 for value in state)
    mask = tuple(
        1 if action_index in {0, (index % 5) + 1} else 0
        for action_index in range(DEFAULT_ACTION_SIZE)
    )
    return {
        "state": state,
        "action_index": (index % 5) + 1,
        "reward": float(index) / 10.0,
        "next_state": next_state,
        "done": bool(index % 2),
        "mask": mask,
    }


def test_replay_buffer_append_evicts_oldest_transition() -> None:
    buffer = ReplayBuffer(capacity=3)

    for index in range(5):
        buffer.append(**make_transition(index))

    assert len(buffer) == 3
    assert [transition.action_index for transition in buffer.transitions()] == [3, 4, 5]
    assert buffer.transitions()[0].state[0] == 2.0


def test_replay_buffer_sample_returns_expected_tensor_shapes() -> None:
    buffer = ReplayBuffer(capacity=5)
    for index in range(5):
        buffer.append(**make_transition(index))

    batch = buffer.sample(4, rng=random.Random(7))

    assert batch.observations.shape == (4, DEFAULT_OBSERVATION_SIZE)
    assert batch.actions.shape == (4,)
    assert batch.rewards.shape == (4,)
    assert batch.next_observations.shape == (4, DEFAULT_OBSERVATION_SIZE)
    assert batch.dones.shape == (4,)
    assert batch.legal_action_masks.shape == (4, DEFAULT_ACTION_SIZE)
    assert batch.observations.dtype == torch.float32
    assert batch.actions.dtype == torch.int64
    assert batch.dones.dtype == torch.bool
    assert batch.legal_action_masks.dtype == torch.bool


def test_replay_buffer_sample_rejects_invalid_batch_sizes() -> None:
    buffer = ReplayBuffer(capacity=2)
    buffer.append(**make_transition(0))

    with pytest.raises(ValueError, match="batch_size must be positive"):
        buffer.sample(0)
    with pytest.raises(ValueError, match="batch_size exceeds replay size"):
        buffer.sample(2)


def test_masked_dqn_forward_matches_frozen_contract_shape() -> None:
    model = MaskedDQN()
    observation_batch = torch.randn(3, DEFAULT_OBSERVATION_SIZE)

    output = model(observation_batch)

    assert output.shape == (3, DEFAULT_ACTION_SIZE)


def test_legal_argmax_selects_highest_value_legal_action() -> None:
    q_values = torch.tensor([1.0, 5.0, 4.0, 9.0])
    mask = (0, 1, 0, 1)

    assert legal_argmax(q_values, mask) == 3


def test_epsilon_greedy_action_only_returns_legal_indices() -> None:
    q_values = torch.tensor([1.0, 5.0, 4.0, 9.0])
    mask = (0, 1, 0, 1)
    rng = random.Random(11)

    actions = {epsilon_greedy_action(q_values, mask, epsilon=1.0, rng=rng) for _ in range(20)}

    assert actions <= {1, 3}
    assert actions


def test_sanitize_mask_falls_back_to_end_turn_for_degenerate_masks() -> None:
    all_illegal = sanitize_legal_action_mask([0] * DEFAULT_ACTION_SIZE)
    wrong_shape = sanitize_legal_action_mask([1, 0, 1], action_size=DEFAULT_ACTION_SIZE)

    assert all_illegal[0].item() is True
    assert int(all_illegal.sum().item()) == 1
    assert wrong_shape[0].item() is True
    assert legal_argmax(torch.tensor([2.0, 9.0, 8.0]), [0, 0, 0]) == 0


def test_checkpoint_round_trip_restores_parameters_and_metadata(tmp_path) -> None:
    model = MaskedDQN()
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.25)

    checkpoint_path = tmp_path / "baseline.pt"
    save_checkpoint(checkpoint_path, model, metadata={"step": 12, "policy": "masked_dqn"})

    restored = MaskedDQN()
    metadata = load_checkpoint(checkpoint_path, restored)

    assert metadata == {"step": 12, "policy": "masked_dqn"}
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(original, loaded)
