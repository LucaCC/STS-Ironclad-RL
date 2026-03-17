# Live Policy Evaluation

## Purpose

This harness runs small batches of live episodes through the same bridge-backed
rollout path used for control-loop testing. It is meant for smoke evaluation,
policy debugging, and quick baseline comparisons against the real game through
CommunicationMod.

It is not a benchmark harness and does not claim simulator-like throughput or
statistical rigor.

## What It Measures

The summary currently reports:

- terminal outcome distribution
- average steps per episode
- policy action distribution
- interruption and failure counts
- average total reward when the bridge exposes numeric `raw_state["reward"]`
- average final score and floor when the final snapshot exposes those fields

Reward and score proxies are opportunistic. If the live bridge does not expose
them, the summary leaves those fields empty.

## Entry Point

Use [`scripts/evaluate_live_policy.py`](/Users/lucacc/Desktop/STS/STS-Ironclad-RL/.worktrees/thread6-eval-harness/scripts/evaluate_live_policy.py).

Example:

```bash
python scripts/evaluate_live_policy.py \
  --transport your_bridge_module:build_transport \
  --policy heuristic \
  --episodes 5 \
  --max-steps 150 \
  --summary-json logs/live_eval/heuristic_summary.json
```

Supported built-in policies:

- `heuristic`
- `random`

Custom policies can be provided as `module:factory` import paths. The factory
must return an object compatible with `sts_ironclad_rl.live.Policy`.

## Transport Assumption

This repository still does not ship a concrete CommunicationMod transport.
The evaluation script therefore requires a `BridgeTransport` factory via
`--transport`. That keeps the evaluation harness lightweight and avoids locking
the repo into one local bridge implementation.

## Safe Usage Guidance

- Keep episode counts small. Live execution is slow and can drift if the game
  window or mod state changes underneath the runner.
- Start with the heuristic policy and 1-3 episodes before trying larger
  batches.
- Prefer a fresh run state and stable game focus while evaluating.
- Treat interruptions such as bridge disconnects, malformed snapshots, and
  max-step exits as control-loop diagnostics, not policy quality signals.
- Do not compare runs as if they were deterministic benchmarks unless the live
  setup and encountered game states are intentionally controlled.

## Design Notes

- Bridge-dependent execution lives in `src/sts_ironclad_rl/live/rollout.py`.
- Pure aggregation and formatting lives in
  `src/sts_ironclad_rl/live/evaluation.py`.
- The CLI only wires transport, policy, runner, and summary output together.
