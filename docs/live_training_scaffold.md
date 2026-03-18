## Live Training Scaffold

The training package now has two coherent layers built on the same live rollout path.

## Collection Layer

`ExperimentRunner` exists for replay-backed collection runs:

- config source: `configs/experiments/*.json`
- artifact root: `artifacts/experiments/<experiment>/<run_id>/`
- outputs: `config.json`, `metadata.json`, `summary.json`, `episodes.jsonl`, `trajectory.jsonl`

## Trainable Baseline Layer

`DQNTrainer` is the first actual trainable baseline:

- config source: `configs/training/masked_dqn_baseline.json`
- artifact root: `artifacts/training/masked_dqn_baseline/`
- outputs: `config.json`, `metrics.jsonl`, `evaluations.jsonl`, `summary.json`, `checkpoints/`

The trainer reuses the shared rollout path, the frozen learner contract, and
the same policy/evaluation helpers used elsewhere in the repo.

## Non-Goals

- no second training stack
- no simulator-first path
- no new DQN-family features in this milestone

## Extension Direction

Future DQN-family work should extend:

- `MaskedDQNConfig` and `MaskedDQN`
- replay sampling behavior
- target computation in `DQNTrainer`

without changing the shared rollout or learner-contract foundations.
