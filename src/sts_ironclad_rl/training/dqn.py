"""Reusable masked-DQN building blocks aligned with the frozen learner contract."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .learner import (
    LEARNER_TRANSITION_SCHEMA_VERSION,
    LearnerActionIndex,
    LearnerObservationLayout,
    LearnerTransition,
)

DEFAULT_OBSERVATION_SIZE = LearnerObservationLayout().vector_size
DEFAULT_ACTION_SIZE = LearnerActionIndex().size
_MASK_FALLBACK_INDEX = 0


@dataclass(frozen=True)
class ReplayBatch:
    """Tensor batch sampled from replay for masked-DQN updates."""

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    dones: Tensor
    legal_action_masks: Tensor


@dataclass
class ReplayBuffer:
    """Bounded replay memory storing learner-contract transitions.

    Stored schema matches ``(state, action_index, reward, next_state, done, mask)`` where
    ``mask`` is the legal-action mask for the next state.
    """

    capacity: int
    _storage: deque[LearnerTransition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._storage = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def append(
        self,
        *,
        state: Sequence[float],
        action_index: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
        mask: Sequence[int],
    ) -> None:
        """Append one transition, evicting the oldest item when at capacity."""

        transition = LearnerTransition(
            schema_version=LEARNER_TRANSITION_SCHEMA_VERSION,
            state=tuple(float(value) for value in state),
            action_index=int(action_index),
            reward=float(reward),
            next_state=tuple(float(value) for value in next_state),
            done=bool(done),
            mask=tuple(int(value) for value in mask),
        )
        _validate_transition(transition)
        self._storage.append(transition)

    def append_transition(self, transition: LearnerTransition) -> None:
        """Append an existing learner transition."""

        _validate_transition(transition)
        self._storage.append(transition)

    def sample(
        self,
        batch_size: int,
        *,
        rng: random.Random | None = None,
        device: torch.device | str | None = None,
    ) -> ReplayBatch:
        """Sample a random batch and materialize it as tensors."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._storage):
            raise ValueError("batch_size exceeds replay size")

        sampler = rng if rng is not None else random
        samples = sampler.sample(list(self._storage), batch_size)
        return ReplayBatch(
            observations=torch.tensor(
                [transition.state for transition in samples],
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.tensor(
                [transition.action_index for transition in samples],
                dtype=torch.int64,
                device=device,
            ),
            rewards=torch.tensor(
                [transition.reward for transition in samples],
                dtype=torch.float32,
                device=device,
            ),
            next_observations=torch.tensor(
                [transition.next_state for transition in samples],
                dtype=torch.float32,
                device=device,
            ),
            dones=torch.tensor(
                [transition.done for transition in samples],
                dtype=torch.bool,
                device=device,
            ),
            legal_action_masks=torch.tensor(
                [transition.mask for transition in samples],
                dtype=torch.bool,
                device=device,
            ),
        )

    def transitions(self) -> tuple[LearnerTransition, ...]:
        """Return a snapshot of stored transitions in insertion order."""

        return tuple(self._storage)


@dataclass(frozen=True)
class MaskedDQNConfig:
    """Simple feed-forward masked-DQN architecture."""

    observation_size: int = DEFAULT_OBSERVATION_SIZE
    action_size: int = DEFAULT_ACTION_SIZE
    hidden_sizes: tuple[int, ...] = (128, 128)

    def __post_init__(self) -> None:
        if self.observation_size <= 0:
            raise ValueError("observation_size must be positive")
        if self.action_size <= 0:
            raise ValueError("action_size must be positive")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain only positive values")


class MaskedDQN(nn.Module):
    """Minimal MLP baseline mapping the 93-dim learner vector to 61 Q-values."""

    def __init__(self, config: MaskedDQNConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskedDQNConfig(
            observation_size=DEFAULT_OBSERVATION_SIZE,
            action_size=DEFAULT_ACTION_SIZE,
        )
        layers: list[nn.Module] = []
        input_size = self.config.observation_size
        for hidden_size in self.config.hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.config.action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        """Return Q-values for one observation or a batch of observations."""

        if observations.ndim not in {1, 2}:
            raise ValueError("observations must be rank 1 or 2")
        if observations.shape[-1] != self.config.observation_size:
            raise ValueError(
                "observation size does not match network input: "
                f"{observations.shape[-1]} != {self.config.observation_size}"
            )
        q_values = self.network(observations.to(dtype=torch.float32))
        return q_values


def legal_argmax(q_values: Tensor | Sequence[float], legal_mask: Tensor | Sequence[int]) -> int:
    """Return the highest-value legal action, falling back to END_TURN on bad masks."""

    q_tensor = _as_q_tensor(q_values)
    mask_tensor = sanitize_legal_action_mask(legal_mask, action_size=q_tensor.shape[0])
    masked_values = q_tensor.masked_fill(~mask_tensor, float("-inf"))
    return int(torch.argmax(masked_values).item())


def epsilon_greedy_action(
    q_values: Tensor | Sequence[float],
    legal_mask: Tensor | Sequence[int],
    *,
    epsilon: float,
    rng: random.Random | None = None,
) -> int:
    """Sample epsilon-greedy actions from the legal set only."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    q_tensor = _as_q_tensor(q_values)
    mask_tensor = sanitize_legal_action_mask(legal_mask, action_size=q_tensor.shape[0])
    legal_indices = torch.nonzero(mask_tensor, as_tuple=False).flatten().tolist()
    if not legal_indices:
        return _safe_fallback_index(q_tensor.shape[0])

    sampler = rng if rng is not None else random
    if sampler.random() < epsilon:
        return int(sampler.choice(legal_indices))
    return legal_argmax(q_tensor, mask_tensor)


def sanitize_legal_action_mask(
    legal_mask: Tensor | Sequence[int],
    *,
    action_size: int = DEFAULT_ACTION_SIZE,
) -> Tensor:
    """Normalize learner masks to a safe fixed-length boolean tensor.

    Any malformed mask, shape mismatch, or all-illegal mask falls back to END_TURN legal only.
    """

    fallback = torch.zeros(action_size, dtype=torch.bool)
    fallback[_safe_fallback_index(action_size)] = True

    try:
        mask_tensor = torch.as_tensor(legal_mask)
    except (TypeError, ValueError):
        return fallback

    if mask_tensor.ndim != 1 or mask_tensor.shape[0] != action_size:
        return fallback

    normalized = mask_tensor.to(dtype=torch.bool)
    if not torch.any(normalized):
        return fallback
    return normalized


def save_checkpoint(
    path: str | Path,
    model: MaskedDQN,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist model weights plus lightweight metadata."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
            "model_config": {
                "observation_size": model.config.observation_size,
                "action_size": model.config.action_size,
                "hidden_sizes": list(model.config.hidden_sizes),
            },
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    model: MaskedDQN,
    *,
    map_location: torch.device | str | None = "cpu",
) -> dict[str, Any]:
    """Load model weights and return checkpoint metadata."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")

    model.load_state_dict(payload["model_state_dict"])
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    return metadata


def _as_q_tensor(q_values: Tensor | Sequence[float]) -> Tensor:
    tensor = torch.as_tensor(q_values, dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError("q_values must be rank 1")
    return tensor


def _validate_transition(transition: LearnerTransition) -> None:
    if len(transition.state) != DEFAULT_OBSERVATION_SIZE:
        raise ValueError(
            "state length does not match learner observation size: "
            f"{len(transition.state)} != {DEFAULT_OBSERVATION_SIZE}"
        )
    if len(transition.next_state) != DEFAULT_OBSERVATION_SIZE:
        raise ValueError(
            "next_state length does not match learner observation size: "
            f"{len(transition.next_state)} != {DEFAULT_OBSERVATION_SIZE}"
        )
    if len(transition.mask) != DEFAULT_ACTION_SIZE:
        raise ValueError(
            "mask length does not match learner action size: "
            f"{len(transition.mask)} != {DEFAULT_ACTION_SIZE}"
        )
    if not 0 <= transition.action_index < DEFAULT_ACTION_SIZE:
        raise ValueError(
            f"action_index does not match learner action bounds: {transition.action_index}"
        )


def _safe_fallback_index(action_size: int) -> int:
    if action_size <= 0:
        raise ValueError("action_size must be positive")
    if action_size == LearnerActionIndex().size:
        return _MASK_FALLBACK_INDEX
    return 0


__all__ = [
    "DEFAULT_ACTION_SIZE",
    "DEFAULT_OBSERVATION_SIZE",
    "MaskedDQN",
    "MaskedDQNConfig",
    "ReplayBatch",
    "ReplayBuffer",
    "epsilon_greedy_action",
    "legal_argmax",
    "load_checkpoint",
    "sanitize_legal_action_mask",
    "save_checkpoint",
]
