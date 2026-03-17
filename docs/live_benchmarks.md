# Live DQN Benchmarks

## Purpose

This workflow makes the first trainable live masked-DQN baseline measurable
against the existing baseline policies without introducing a second rollout or
evaluation stack.

All policy evaluation still runs through:

- `scripts/_live_utils.py` runner setup
- `src/sts_ironclad_rl/live/rollout.py`
- `src/sts_ironclad_rl/live/evaluation.py`

The benchmark layer only adds config loading, comparison reporting, and
artifact writing around that shared path.

## Entry Points

Baseline comparison:

```bash
python scripts/run_live_benchmark.py \
  --transport your_bridge_module:build_transport \
  --config configs/benchmarks/baseline_eval.json
```

First DQN training run:

```bash
python scripts/train_live_dqn.py \
  --transport your_bridge_module:build_transport \
  --config configs/training/live_dqn_benchmark.json \
  --output-dir artifacts/training/live_dqn_benchmark
```

Post-training DQN vs baselines:

```bash
python scripts/run_live_benchmark.py \
  --transport your_bridge_module:build_transport \
  --config configs/benchmarks/dqn_vs_baselines.json
```

If the checkpoint path in
`configs/benchmarks/dqn_vs_baselines.json` does not match your output
directory, update the `dqn_checkpoint:...` policy reference first.

Single-policy evaluation still works through the shared evaluation CLI:

```bash
python scripts/evaluate_live_policy.py \
  --transport your_bridge_module:build_transport \
  --policy dqn_checkpoint:artifacts/training/live_dqn_benchmark/checkpoints/checkpoint_final.pt \
  --policy-name masked_dqn \
  --episodes 3 \
  --max-steps 150
```

## Configs

- `configs/benchmarks/baseline_eval.json`: random legal plus simple heuristic
- `configs/training/live_dqn_benchmark.json`: first reproducible trainer config
- `configs/benchmarks/dqn_vs_baselines.json`: post-training comparison config

## Artifacts

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

## Metrics To Trust First

For early live training, prioritize:

- interruption rate: catches bridge and control-loop instability
- terminal rate: shows whether episodes are completing at all
- victory/defeat counts: use as a weak directional signal, not a strong claim
- mean steps: useful for spotting degenerate loops or stalled play
- mean total reward: only if your bridge reward signal is populated consistently
- invalid action and mask fallback counts: DQN safety diagnostics
- trainer loss, epsilon, and optimization step count: tells you whether learning
  is actually updating rather than only collecting

Final score and floor are useful secondary context when the live bridge exposes
them, but they should not outrank completion and interruption metrics during
early integration.

## Conservative Interpretation

- Live execution is slow, so episode counts stay small and variance stays high.
- Different runs can see different combats, rewards, and game states.
- A few extra wins over `random_legal` are encouraging, not conclusive.
- Be cautious comparing DQN against `simple_heuristic`; the heuristic is stable
  and deterministic while the DQN remains sensitive to data sparsity.
- If interruption rate is high, fix the bridge or rollout stability first. Do
  not treat policy outcome deltas as meaningful until that control path is
  reasonably stable.

## Recommended First Experiment Plan

1. Run `baseline_eval.json` to record random legal and heuristic behavior.
2. Train one small DQN run with `live_dqn_benchmark.json`.
3. Inspect `artifacts/training/.../summary.json` and `evaluations.jsonl` to
   confirm updates are happening and greedy eval is not regressing badly.
4. Update the DQN checkpoint path in `dqn_vs_baselines.json` if needed.
5. Run the post-training benchmark and compare the `comparison.txt` output.
