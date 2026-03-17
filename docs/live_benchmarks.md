# Live Benchmarks

## Purpose

This workflow measures the current masked-DQN baseline against the existing
baseline policies without introducing a second evaluation stack.

All policy execution still runs through:

- `sts_ironclad_rl.live.build_live_episode_runner`
- `sts_ironclad_rl.live.PolicyEvaluator`
- `sts_ironclad_rl.live.LiveEpisodeRunner`

The benchmark layer only adds config loading, artifact writing, and
side-by-side reporting.

## Canonical Commands

Baseline benchmark:

```bash
python scripts/run_live_benchmark.py \
  --transport your_bridge_module:build_transport \
  --config configs/benchmarks/baseline_eval.json
```

Baseline training:

```bash
python scripts/train_live_dqn.py \
  --transport your_bridge_module:build_transport \
  --config configs/training/masked_dqn_baseline.json \
  --output-dir artifacts/training/masked_dqn_baseline
```

Checkpoint evaluation:

```bash
python scripts/evaluate_live_policy.py \
  --transport your_bridge_module:build_transport \
  --policy dqn_checkpoint:artifacts/training/masked_dqn_baseline/checkpoints/checkpoint_final.pt \
  --policy-name masked_dqn \
  --episodes 3
```

Post-training comparison:

```bash
python scripts/run_live_benchmark.py \
  --transport your_bridge_module:build_transport \
  --config configs/benchmarks/masked_dqn_vs_baselines.json
```

## Canonical Artifacts

Training writes:

- `config.json`
- `metrics.jsonl`
- `evaluations.jsonl`
- `summary.json`
- `checkpoints/checkpoint_final.pt`

Benchmark runs write:

- `config.json`
- `summaries.json`
- `comparison.json`
- `comparison.txt`

Benchmark reporting reads trainer summaries through the canonical checkpoint
layout instead of hardcoded ad hoc path math in each script.

## Metrics To Trust First

- interruption rate
- terminal rate
- win/loss counts as a weak directional signal
- mean steps
- invalid action count
- mask fallback count
- epsilon and optimization step count from the trainer summary

Treat score, floor, and reward as secondary context unless the live bridge is
exposing them reliably.

## Conservative Interpretation

- live episode counts remain small
- combat draws are not controlled enough for strong causal claims
- beating `random_legal` is a useful milestone, not a research conclusion
- `simple_heuristic` remains a stronger and more stable baseline than random legal
