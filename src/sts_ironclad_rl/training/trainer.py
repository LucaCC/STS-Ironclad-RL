"""Masked-DQN trainer loop built on the shared live rollout path."""

from __future__ import annotations

import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor, nn

from ..live import (
    ActionDecision,
    EncodedObservation,
    EndTurnAction,
    EvaluationCase,
    EvaluationSummary,
    PolicyEvaluator,
    RolloutResult,
    RolloutRunner,
    summary_to_dict,
)
from .dqn import (
    MaskedDQN,
    MaskedDQNConfig,
    ReplayBatch,
    ReplayBuffer,
    epsilon_greedy_action,
    sanitize_legal_action_mask,
)
from .learner import LearnerActionIndex, LearnerObservationEncoder, LearnerTransitionExtractor


@dataclass(frozen=True)
class EpsilonSchedule:
    """Linear epsilon schedule for exploration during collection."""

    initial: float = 1.0
    final: float = 0.05
    decay_steps: int = 10_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.final <= 1.0:
            raise ValueError("final epsilon must be between 0 and 1")
        if not 0.0 <= self.initial <= 1.0:
            raise ValueError("initial epsilon must be between 0 and 1")
        if self.final > self.initial:
            raise ValueError("final epsilon must be less than or equal to initial epsilon")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")

    def value(self, step: int) -> float:
        """Return the scheduled epsilon for the provided environment step."""

        if step <= 0:
            return self.initial
        progress = min(float(step) / float(self.decay_steps), 1.0)
        return self.initial + ((self.final - self.initial) * progress)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "initial": self.initial,
            "final": self.final,
            "decay_steps": self.decay_steps,
        }


@dataclass(frozen=True)
class DQNTrainerConfig:
    """Configuration for a minimal live masked-DQN baseline."""

    train_episodes: int = 20
    evaluation_episodes: int = 3
    max_steps_per_episode: int = 200
    replay_buffer_size: int = 10_000
    batch_size: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_schedule: EpsilonSchedule = field(default_factory=EpsilonSchedule)
    target_update_frequency: int = 100
    warmup_steps: int = 1_000
    evaluation_cadence: int = 10
    checkpoint_cadence: int = 10
    gradient_clip_norm: float | None = 10.0
    metric_window_size: int = 20
    seed: int | None = None
    network: MaskedDQNConfig = field(default_factory=MaskedDQNConfig)
    train_case_name: str = "dqn_train"
    evaluation_case_name: str = "dqn_eval"

    def __post_init__(self) -> None:
        if self.train_episodes <= 0:
            raise ValueError("train_episodes must be positive")
        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")
        if self.replay_buffer_size <= 0:
            raise ValueError("replay_buffer_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size > self.replay_buffer_size:
            raise ValueError("batch_size must not exceed replay_buffer_size")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if self.target_update_frequency <= 0:
            raise ValueError("target_update_frequency must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.evaluation_cadence <= 0:
            raise ValueError("evaluation_cadence must be positive")
        if self.checkpoint_cadence <= 0:
            raise ValueError("checkpoint_cadence must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if self.metric_window_size <= 0:
            raise ValueError("metric_window_size must be positive")
        if not self.train_case_name.strip():
            raise ValueError("train_case_name must not be empty")
        if not self.evaluation_case_name.strip():
            raise ValueError("evaluation_case_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_episodes": self.train_episodes,
            "evaluation_episodes": self.evaluation_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "replay_buffer_size": self.replay_buffer_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_schedule": self.epsilon_schedule.to_dict(),
            "target_update_frequency": self.target_update_frequency,
            "warmup_steps": self.warmup_steps,
            "evaluation_cadence": self.evaluation_cadence,
            "checkpoint_cadence": self.checkpoint_cadence,
            "gradient_clip_norm": self.gradient_clip_norm,
            "metric_window_size": self.metric_window_size,
            "seed": self.seed,
            "network": {
                "observation_size": self.network.observation_size,
                "action_size": self.network.action_size,
                "hidden_sizes": list(self.network.hidden_sizes),
            },
            "train_case_name": self.train_case_name,
            "evaluation_case_name": self.evaluation_case_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DQNTrainerConfig:
        """Build a validated trainer config from a raw JSON-like payload."""

        epsilon_payload = payload.get("epsilon_schedule", {})
        if not isinstance(epsilon_payload, dict):
            raise ValueError("epsilon_schedule must be a JSON object")
        network_payload = payload.get("network", {})
        if not isinstance(network_payload, dict):
            raise ValueError("network must be a JSON object")

        return cls(
            train_episodes=int(payload.get("train_episodes", 20)),
            evaluation_episodes=int(payload.get("evaluation_episodes", 3)),
            max_steps_per_episode=int(payload.get("max_steps_per_episode", 200)),
            replay_buffer_size=int(payload.get("replay_buffer_size", 10_000)),
            batch_size=int(payload.get("batch_size", 32)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            gamma=float(payload.get("gamma", 0.99)),
            epsilon_schedule=EpsilonSchedule(
                initial=float(epsilon_payload.get("initial", 1.0)),
                final=float(epsilon_payload.get("final", 0.05)),
                decay_steps=int(epsilon_payload.get("decay_steps", 10_000)),
            ),
            target_update_frequency=int(payload.get("target_update_frequency", 100)),
            warmup_steps=int(payload.get("warmup_steps", 1_000)),
            evaluation_cadence=int(payload.get("evaluation_cadence", 10)),
            checkpoint_cadence=int(payload.get("checkpoint_cadence", 10)),
            gradient_clip_norm=(
                None
                if payload.get("gradient_clip_norm") is None
                else float(payload["gradient_clip_norm"])
            ),
            metric_window_size=int(payload.get("metric_window_size", 20)),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
            network=MaskedDQNConfig(
                observation_size=int(network_payload.get("observation_size", 93)),
                action_size=int(network_payload.get("action_size", 61)),
                hidden_sizes=tuple(
                    int(size) for size in network_payload.get("hidden_sizes", (128, 128))
                ),
            ),
            train_case_name=str(payload.get("train_case_name", "dqn_train")),
            evaluation_case_name=str(payload.get("evaluation_case_name", "dqn_eval")),
        )


@dataclass(frozen=True)
class PolicySelectionStats:
    """Episode-local action-selection diagnostics."""

    mask_fallback_count: int = 0
    invalid_action_count: int = 0


@dataclass(frozen=True)
class TrainingEpisodeMetrics:
    """Per-episode trainer metrics for logging and summaries."""

    episode_index: int
    environment_steps: int
    optimization_steps: int
    replay_size: int
    epsilon: float
    transition_count: int
    episode_return: float
    average_reward: float
    average_loss: float | None
    episode_length: int
    outcome: str | None
    terminal: bool
    total_reward_proxy: float | None
    mask_fallback_count: int
    invalid_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainerState:
    """Mutable training progress serialized in checkpoints."""

    completed_episodes: int = 0
    environment_steps: int = 0
    optimization_steps: int = 0
    target_sync_steps: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DQNTrainingResult:
    """End-of-run trainer result."""

    state: TrainerState
    episode_metrics: tuple[TrainingEpisodeMetrics, ...]
    evaluation_summaries: tuple[EvaluationSummary, ...]
    output_dir: Path | None = None


@dataclass(frozen=True)
class MaskedDQNPolicy:
    """Policy adapter from a Q-network to the shared live `Policy` contract."""

    model: MaskedDQN
    epsilon: float
    device: torch.device
    name: str = "masked_dqn"
    observation_encoder: LearnerObservationEncoder = field(
        default_factory=LearnerObservationEncoder
    )
    action_index: LearnerActionIndex = field(default_factory=LearnerActionIndex)
    rng: random.Random = field(default_factory=random.Random, repr=False)
    _mask_fallback_count: int = field(init=False, default=0, repr=False)
    _invalid_action_count: int = field(init=False, default=0, repr=False)

    def select_action(self, observation: EncodedObservation) -> ActionDecision:
        state_vector = self.observation_encoder.encode(observation).vector
        raw_mask = self.action_index.legal_mask(observation)
        safe_mask = sanitize_legal_action_mask(raw_mask, action_size=self.action_index.size)
        if tuple(int(value) for value in safe_mask.tolist()) != tuple(raw_mask):
            object.__setattr__(self, "_mask_fallback_count", self._mask_fallback_count + 1)

        with torch.no_grad():
            q_values = self.model(
                torch.tensor(state_vector, dtype=torch.float32, device=self.device)
            ).detach()
        safe_mask = safe_mask.to(device=q_values.device)
        selected_index = epsilon_greedy_action(
            q_values,
            safe_mask,
            epsilon=self.epsilon,
            rng=self.rng,
        )
        action_id = self.action_index.index_to_action_id(selected_index)
        if action_id not in observation.legal_action_ids:
            object.__setattr__(self, "_invalid_action_count", self._invalid_action_count + 1)
            action_id = _fallback_action_id(observation.legal_action_ids)

        return ActionDecision(
            action_id=action_id,
            metadata={"action_index": selected_index, "epsilon": self.epsilon},
        )

    def stats(self) -> PolicySelectionStats:
        return PolicySelectionStats(
            mask_fallback_count=self._mask_fallback_count,
            invalid_action_count=self._invalid_action_count,
        )


@dataclass
class DQNTrainer:
    """Collect live episodes, build replay, and optimize a masked-DQN baseline."""

    rollout_runner: RolloutRunner
    config: DQNTrainerConfig = field(default_factory=DQNTrainerConfig)
    evaluation_runner: RolloutRunner | None = None
    transition_extractor: LearnerTransitionExtractor = field(
        default_factory=LearnerTransitionExtractor
    )
    device: torch.device | str = "cpu"
    online_network: MaskedDQN | None = None
    target_network: MaskedDQN | None = None
    replay_buffer: ReplayBuffer | None = None

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self._rng = random.Random(self.config.seed)
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        self.online_network = self.online_network or MaskedDQN(self.config.network)
        self.target_network = self.target_network or MaskedDQN(self.config.network)
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_fn: nn.Module = nn.MSELoss()
        self.replay_buffer = self.replay_buffer or ReplayBuffer(self.config.replay_buffer_size)
        self.state = TrainerState()
        self._episode_metrics: list[TrainingEpisodeMetrics] = []
        self._evaluation_summaries: list[EvaluationSummary] = []
        self._recent_metrics: deque[TrainingEpisodeMetrics] = deque(
            maxlen=self.config.metric_window_size
        )

    def train(self, *, output_dir: Path | None = None) -> DQNTrainingResult:
        """Run the configured number of live episodes through the trainer loop."""

        output_paths = _TrainerOutputPaths.create(output_dir) if output_dir is not None else None
        if output_paths is not None:
            _write_json(output_paths.config_path, self.config.to_dict())

        for _ in range(self.config.train_episodes):
            episode_index = self.state.completed_episodes
            epsilon = self.config.epsilon_schedule.value(self.state.environment_steps)
            policy = MaskedDQNPolicy(
                model=self.online_network,
                epsilon=epsilon,
                device=self.device,
                rng=self._rng,
            )
            result = self.rollout_runner.run_episode(
                policy=policy,
                evaluation_case=EvaluationCase(
                    name=self.config.train_case_name,
                    max_steps=self.config.max_steps_per_episode,
                ),
            )
            metrics = self._consume_rollout(
                episode_index=episode_index,
                epsilon=epsilon,
                result=result,
                policy_stats=policy.stats(),
            )
            self._episode_metrics.append(metrics)
            self._recent_metrics.append(metrics)
            self.state = TrainerState(
                completed_episodes=self.state.completed_episodes + 1,
                environment_steps=self.state.environment_steps,
                optimization_steps=self.state.optimization_steps,
                target_sync_steps=self.state.target_sync_steps,
            )

            if output_paths is not None:
                _append_jsonl(output_paths.metrics_path, metrics.to_dict())
                _write_json(output_paths.summary_path, self.training_summary())

            if self.state.completed_episodes % self.config.evaluation_cadence == 0:
                evaluation_summary = self.evaluate()
                self._evaluation_summaries.append(evaluation_summary)
                if output_paths is not None:
                    _append_jsonl(
                        output_paths.evaluations_path, summary_to_dict(evaluation_summary)
                    )

            if output_paths is not None and (
                self.state.completed_episodes % self.config.checkpoint_cadence == 0
            ):
                self.save_checkpoint(
                    output_paths.checkpoints_dir
                    / f"checkpoint_ep{self.state.completed_episodes:06d}.pt"
                )

        if output_paths is not None:
            self.save_checkpoint(output_paths.checkpoints_dir / "checkpoint_final.pt")
            _write_json(output_paths.summary_path, self.training_summary())

        return DQNTrainingResult(
            state=self.state,
            episode_metrics=tuple(self._episode_metrics),
            evaluation_summaries=tuple(self._evaluation_summaries),
            output_dir=output_dir,
        )

    def evaluate(self) -> EvaluationSummary:
        """Run a greedy evaluation batch through the shared rollout path."""

        evaluator = PolicyEvaluator(runner=self.evaluation_runner or self.rollout_runner)
        policy = MaskedDQNPolicy(
            model=self.online_network,
            epsilon=0.0,
            device=self.device,
            name="masked_dqn_eval",
            rng=random.Random(self.config.seed),
        )
        summary = evaluator.evaluate(
            policy=policy,
            episode_count=self.config.evaluation_episodes,
            evaluation_case=EvaluationCase(
                name=self.config.evaluation_case_name,
                max_steps=self.config.max_steps_per_episode,
            ),
        ).summary
        stats = policy.stats()
        return EvaluationSummary(
            policy_name=summary.policy_name,
            case_name=summary.case_name,
            episode_count=summary.episode_count,
            terminal_episode_count=summary.terminal_episode_count,
            interruption_count=summary.interruption_count,
            outcome_counts=summary.outcome_counts,
            failure_counts=summary.failure_counts,
            action_counts=summary.action_counts,
            mean_steps=summary.mean_steps,
            mean_total_reward=summary.mean_total_reward,
            mean_final_score=summary.mean_final_score,
            mean_final_floor=summary.mean_final_floor,
            metadata={
                **dict(summary.metadata),
                "invalid_action_count": stats.invalid_action_count,
                "mask_fallback_count": stats.mask_fallback_count,
            },
        )

    def training_summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable training summary."""

        recent_summary = summarize_training_metrics(tuple(self._recent_metrics))
        return {
            "state": self.state.to_dict(),
            "replay_size": len(self.replay_buffer),
            "recent_metrics": recent_summary,
            "evaluation_count": len(self._evaluation_summaries),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist trainer state, optimizer state, and replay contents."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "trainer_state": self.state.to_dict(),
                "online_model_state_dict": self.online_network.state_dict(),
                "target_model_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "replay": [transition_to_dict(item) for item in self.replay_buffer.transitions()],
                "metadata": {
                    **self.training_summary(),
                    **dict(metadata or {}),
                },
            },
            checkpoint_path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: torch.device | str | None = "cpu",
    ) -> dict[str, Any]:
        """Restore trainer progress from a checkpoint and return its metadata."""

        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a dictionary")

        self.online_network.load_state_dict(payload["online_model_state_dict"])
        self.target_network.load_state_dict(payload["target_model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        trainer_state = payload.get("trainer_state")
        if not isinstance(trainer_state, dict):
            raise ValueError("checkpoint trainer_state must be a dictionary")
        self.state = TrainerState(
            completed_episodes=int(trainer_state["completed_episodes"]),
            environment_steps=int(trainer_state["environment_steps"]),
            optimization_steps=int(trainer_state["optimization_steps"]),
            target_sync_steps=int(trainer_state["target_sync_steps"]),
        )

        replay_items = payload.get("replay", [])
        if not isinstance(replay_items, list):
            raise ValueError("checkpoint replay must be a list")
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        for item in replay_items:
            self.replay_buffer.append(**item)

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be a dictionary")
        return metadata

    def _consume_rollout(
        self,
        *,
        episode_index: int,
        epsilon: float,
        result: RolloutResult,
        policy_stats: PolicySelectionStats,
    ) -> TrainingEpisodeMetrics:
        transitions = self.transition_extractor.extract(result.entries)
        losses: list[float] = []
        for transition in transitions:
            self.replay_buffer.append_transition(transition)
            self.state = TrainerState(
                completed_episodes=self.state.completed_episodes,
                environment_steps=self.state.environment_steps + 1,
                optimization_steps=self.state.optimization_steps,
                target_sync_steps=self.state.target_sync_steps,
            )
            loss = self._maybe_optimize()
            if loss is not None:
                losses.append(loss)

        episode_return = float(sum(transition.reward for transition in transitions))
        average_reward = episode_return / len(transitions) if transitions else 0.0
        return TrainingEpisodeMetrics(
            episode_index=episode_index,
            environment_steps=self.state.environment_steps,
            optimization_steps=self.state.optimization_steps,
            replay_size=len(self.replay_buffer),
            epsilon=epsilon,
            transition_count=len(transitions),
            episode_return=episode_return,
            average_reward=average_reward,
            average_loss=fmean(losses) if losses else None,
            episode_length=result.step_count,
            outcome=result.outcome,
            terminal=result.terminal,
            total_reward_proxy=result.total_reward,
            mask_fallback_count=policy_stats.mask_fallback_count,
            invalid_action_count=policy_stats.invalid_action_count,
        )

    def _maybe_optimize(self) -> float | None:
        if self.state.environment_steps < self.config.warmup_steps:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(
            self.config.batch_size,
            rng=self._rng,
            device=self.device,
        )
        loss = self._optimize_batch(batch)
        next_optimization_steps = self.state.optimization_steps + 1
        next_target_syncs = self.state.target_sync_steps
        if should_sync_target_network(
            optimization_steps=next_optimization_steps,
            target_update_frequency=self.config.target_update_frequency,
        ):
            self.target_network.load_state_dict(self.online_network.state_dict())
            next_target_syncs += 1
        self.state = TrainerState(
            completed_episodes=self.state.completed_episodes,
            environment_steps=self.state.environment_steps,
            optimization_steps=next_optimization_steps,
            target_sync_steps=next_target_syncs,
        )
        return loss

    def _optimize_batch(self, batch: ReplayBatch) -> float:
        self.online_network.train()
        q_values = self.online_network(batch.observations)
        chosen_q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(batch.next_observations)
            next_action_values = _masked_next_q_values(
                next_q_values,
                batch.legal_action_masks,
            )
            targets = batch.rewards + (
                (~batch.dones).to(dtype=torch.float32) * self.config.gamma * next_action_values
            )

        loss = self.loss_fn(chosen_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.online_network.parameters(),
                max_norm=self.config.gradient_clip_norm,
            )
        self.optimizer.step()
        self.online_network.eval()
        return float(loss.item())


@dataclass(frozen=True)
class _TrainerOutputPaths:
    root_dir: Path
    config_path: Path
    metrics_path: Path
    evaluations_path: Path
    summary_path: Path
    checkpoints_dir: Path

    @classmethod
    def create(cls, root_dir: Path) -> _TrainerOutputPaths:
        root_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = root_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            root_dir=root_dir,
            config_path=root_dir / "config.json",
            metrics_path=root_dir / "metrics.jsonl",
            evaluations_path=root_dir / "evaluations.jsonl",
            summary_path=root_dir / "summary.json",
            checkpoints_dir=checkpoints_dir,
        )


def should_sync_target_network(
    *,
    optimization_steps: int,
    target_update_frequency: int,
) -> bool:
    """Return whether a hard target-network sync is due on this optimizer step."""

    if optimization_steps <= 0:
        return False
    if target_update_frequency <= 0:
        raise ValueError("target_update_frequency must be positive")
    return optimization_steps % target_update_frequency == 0


def summarize_training_metrics(
    metrics: tuple[TrainingEpisodeMetrics, ...],
) -> dict[str, Any]:
    """Aggregate recent episode metrics into a compact summary payload."""

    if not metrics:
        return {
            "episode_count": 0,
            "average_loss": None,
            "average_reward": None,
            "average_episode_return": None,
            "average_episode_length": None,
            "epsilon": None,
            "outcomes": {},
            "mask_fallback_count": 0,
            "invalid_action_count": 0,
        }

    average_losses = [item.average_loss for item in metrics if item.average_loss is not None]
    return {
        "episode_count": len(metrics),
        "average_loss": fmean(average_losses) if average_losses else None,
        "average_reward": fmean(item.average_reward for item in metrics),
        "average_episode_return": fmean(item.episode_return for item in metrics),
        "average_episode_length": fmean(float(item.episode_length) for item in metrics),
        "epsilon": metrics[-1].epsilon,
        "outcomes": dict(sorted(Counter(item.outcome or "unknown" for item in metrics).items())),
        "mask_fallback_count": sum(item.mask_fallback_count for item in metrics),
        "invalid_action_count": sum(item.invalid_action_count for item in metrics),
    }


def transition_to_dict(transition) -> dict[str, Any]:
    """Return a JSON- and checkpoint-serializable learner transition payload."""

    return {
        "state": list(transition.state),
        "action_index": transition.action_index,
        "reward": transition.reward,
        "next_state": list(transition.next_state),
        "done": transition.done,
        "mask": list(transition.mask),
    }


def load_dqn_trainer_config(config_path: str | Path) -> DQNTrainerConfig:
    """Load one trainer config from JSON."""

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trainer config must be a JSON object")
    return DQNTrainerConfig.from_dict(payload)


def load_trained_dqn_policy(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    epsilon: float = 0.0,
    policy_name: str = "masked_dqn",
    seed: int | None = None,
) -> MaskedDQNPolicy:
    """Load a trained DQN checkpoint into the shared live-policy adapter."""

    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")

    network_config = _checkpoint_network_config(payload)
    model = MaskedDQN(network_config)
    state_dict = _checkpoint_state_dict(payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return MaskedDQNPolicy(
        model=model,
        epsilon=epsilon,
        device=torch.device(device),
        name=policy_name,
        rng=random.Random(seed),
    )


def _masked_next_q_values(q_values: Tensor, legal_action_masks: Tensor) -> Tensor:
    safe_masks = _sanitize_batch_masks(legal_action_masks, action_size=q_values.shape[1])
    masked_q_values = q_values.masked_fill(~safe_masks, float("-inf"))
    return masked_q_values.max(dim=1).values


def _sanitize_batch_masks(legal_action_masks: Tensor, *, action_size: int) -> Tensor:
    masks = legal_action_masks.to(dtype=torch.bool)
    if masks.ndim != 2 or masks.shape[1] != action_size:
        raise ValueError("legal_action_masks must be rank 2 with one row per action vector")
    sanitized = masks.clone()
    invalid_rows = ~sanitized.any(dim=1)
    sanitized[invalid_rows, 0] = True
    return sanitized


def _fallback_action_id(legal_action_ids: tuple[str, ...]) -> str:
    if EndTurnAction().action_id in legal_action_ids:
        return EndTurnAction().action_id
    if not legal_action_ids:
        raise ValueError("legal_action_ids must not be empty")
    return legal_action_ids[0]


def _checkpoint_network_config(payload: dict[str, Any]) -> MaskedDQNConfig:
    if "config" in payload and isinstance(payload["config"], dict):
        raw_network = payload["config"].get("network")
        if isinstance(raw_network, dict):
            return MaskedDQNConfig(
                observation_size=int(raw_network["observation_size"]),
                action_size=int(raw_network["action_size"]),
                hidden_sizes=tuple(int(size) for size in raw_network["hidden_sizes"]),
            )

    if "model_config" in payload and isinstance(payload["model_config"], dict):
        raw_network = payload["model_config"]
        return MaskedDQNConfig(
            observation_size=int(raw_network["observation_size"]),
            action_size=int(raw_network["action_size"]),
            hidden_sizes=tuple(int(size) for size in raw_network["hidden_sizes"]),
        )

    return MaskedDQNConfig()


def _checkpoint_state_dict(payload: dict[str, Any]) -> dict[str, Tensor]:
    if "online_model_state_dict" in payload and isinstance(
        payload["online_model_state_dict"], dict
    ):
        return payload["online_model_state_dict"]
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    raise ValueError("checkpoint is missing a supported model state dict")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DQNTrainer",
    "DQNTrainerConfig",
    "DQNTrainingResult",
    "EpsilonSchedule",
    "MaskedDQNPolicy",
    "PolicySelectionStats",
    "TrainerState",
    "TrainingEpisodeMetrics",
    "load_dqn_trainer_config",
    "load_trained_dqn_policy",
    "should_sync_target_network",
    "summarize_training_metrics",
    "transition_to_dict",
]
