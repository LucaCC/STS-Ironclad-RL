## Architecture

The repository now has one coherent live-game-first learning stack.

## Primary Path

The canonical execution path is:

`LiveGameBridge -> LiveEpisodeRunner -> ReplayEntry -> LearnerTransitionExtractor -> ReplayBuffer -> MaskedDQN/DQNTrainer -> checkpoint -> PolicyEvaluator -> benchmark report`

Evaluation, data collection, and training all reuse the same live rollout
contract. There is no second simulator-first or Gym-first execution loop.

## Package Layout

- `src/sts_ironclad_rl/integration/`
  Bridge protocol types, transport boundary, session lifecycle, and raw trace logging.
- `src/sts_ironclad_rl/live/`
  Observation encoding, action contracts, baseline policies, rollout execution,
  evaluation helpers, and shared CLI wiring for live entrypoints.
- `src/sts_ironclad_rl/training/`
  Frozen learner contract, masked-DQN baseline, trainer, experiment specs,
  artifact layouts, and benchmark comparison helpers.
- `src/sts_ironclad_rl/env/`
  Narrow deterministic support code kept for tests and offline reasoning.

## Current Baseline Components

- `LearnerObservationEncoder` and `LearnerActionIndex`
  freeze the 93-feature observation vector and 61-action learner space.
- `LearnerTransitionExtractor`
  converts shared live replay entries into `(state, action_index, reward, next_state, done, mask)`.
- `ReplayBuffer`, `MaskedDQN`, and `DQNTrainer`
  provide the first replay-backed trainable baseline.
- `PolicyEvaluator` and `BenchmarkArtifactStore`
  evaluate baseline and checkpoint-backed policies through the same rollout path.

## Artifact Conventions

- collection experiments:
  `artifacts/experiments/<experiment_slug>/<run_id>/`
- masked-DQN training:
  `artifacts/training/<run_name>/`
- benchmark comparisons:
  `artifacts/benchmarks/<experiment_slug>/<run_id>/`

Canonical masked-DQN training artifacts are:

- `config.json`
- `metrics.jsonl`
- `evaluations.jsonl`
- `summary.json`
- `checkpoints/checkpoint_final.pt`

Benchmark code derives trainer summaries from that canonical checkpoint layout
instead of duplicating path logic in scripts.

## Design Constraints

- keep the rollout-first architecture
- keep one training/evaluation stack
- preserve extension points for Double DQN, dueling heads, prioritized replay,
  and related Rainbow-like upgrades without implementing them yet
- keep docs and configs honest about live benchmark limits
