# Live Policy Evaluation

## Purpose

Use `scripts/evaluate_live_policy.py` for direct policy checks through the
shared live rollout path.

This is the quick inspection tool. It is not the benchmark harness.

## Supported Policies

- `simple_heuristic`
- `random_legal`
- `dqn_checkpoint:/path/to/checkpoint.pt`
- `module:factory` for custom policies

Checkpoint-backed DQN evaluation uses the same live policy adapter and frozen
learner contract as training and benchmarking.

## Example

```bash
python scripts/evaluate_live_policy.py \
  --transport your_bridge_module:build_transport \
  --policy dqn_checkpoint:artifacts/training/masked_dqn_baseline/checkpoints/checkpoint_final.pt \
  --policy-name masked_dqn \
  --episodes 3 \
  --max-steps 150
```

## Notes

- keep episode counts small
- treat interruptions as control-path diagnostics first
- prefer the benchmark workflow when you need side-by-side policy comparison
