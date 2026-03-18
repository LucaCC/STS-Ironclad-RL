# Live-Game-First Milestone

This document now tracks the integrated result of the live-game-first work,
not the original planning state.

## Implemented Now

- live bridge-facing rollout via `LiveEpisodeRunner`
- shared replay schema for every live episode path
- baseline live policies: `random_legal` and `simple_heuristic`
- frozen learner contract on top of live replay
- replay-backed masked-DQN baseline trainer with evaluation and checkpointing
- benchmark comparison flow over baselines and checkpoint-backed DQN policies

## Canonical Workflow

1. Run `configs/benchmarks/baseline_eval.json` to capture baseline behavior.
2. Train with `configs/training/masked_dqn_baseline.json`.
3. Inspect `artifacts/training/masked_dqn_baseline/summary.json`,
   `metrics.jsonl`, and `evaluations.jsonl`.
4. Evaluate the final checkpoint directly if needed.
5. Run `configs/benchmarks/masked_dqn_vs_baselines.json` to compare the
   checkpoint against the baselines.

## What Success Means At This Stage

Success for this milestone is not strong game performance. It is:

- one repeatable end-to-end training path that actually updates a learner
- coherent artifacts and configs for repeated live experiments
- deterministic non-live tests for the learner, replay, trainer, and reporting code
- documentation that matches the code and calls out the live benchmark limits clearly

## Limits That Still Matter

- no concrete transport implementation is shipped in-repo
- live runs are low-throughput and high-variance
- benchmarks are integration-grade, not publication-grade
- only the first masked-DQN baseline is implemented

## Extension Points Kept Clean

The current layout intentionally leaves room for:

- alternative target computation such as Double DQN
- richer Q-network heads such as dueling architectures
- replay policy changes such as prioritized replay

Those upgrades should layer onto the current learner/trainer interfaces instead
of creating a parallel stack.
