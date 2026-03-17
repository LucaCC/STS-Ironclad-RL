# Trainable-Agent Milestone

## Current Capabilities

- live bridge-backed episode rollout
- replay extraction into the frozen learner contract
- masked-DQN baseline with replay, target network, epsilon schedule, metrics,
  checkpointing, and greedy evaluation
- baseline and DQN evaluation through one shared benchmark path

## Current Constraints

- no in-repo transport implementation
- no simulator-first training path
- no Rainbow features yet
- live experiments remain small, slow, and noisy

## Recommended First Experiment Loop

1. Run `configs/benchmarks/baseline_eval.json`.
2. Train with `configs/training/masked_dqn_baseline.json`.
3. Inspect training artifacts under `artifacts/training/masked_dqn_baseline/`.
4. Evaluate `checkpoints/checkpoint_final.pt` directly if needed.
5. Run `configs/benchmarks/masked_dqn_vs_baselines.json`.

## What To Look For

- low interruption rate
- nonzero optimizer updates after warmup
- sane mask fallback and invalid-action counts
- no obvious regression in greedy evaluation summaries
- directional improvement over `random_legal` before claiming anything stronger

## Likely Next Upgrades

- Double DQN target selection
- dueling network heads
- prioritized replay
- more systematic evaluation controls once live execution is stable enough
