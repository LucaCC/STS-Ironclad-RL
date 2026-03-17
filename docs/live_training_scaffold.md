## Live Training Scaffold

This scaffold turns the existing live rollout path into a lightweight
experimentation substrate for data collection.

### What It Does Today

- defines explicit `ExperimentSpec` configs for live collection runs
- runs repeated episodes through the existing `RolloutRunner` contract
- writes stable run artifacts under `artifacts/experiments/<experiment>/<run_id>/`
- saves replay-backed `trajectory.jsonl` and per-episode `episodes.jsonl`
- records `config.json`, `metadata.json`, and `summary.json` for reproducibility
- leaves policy selection open so current `simple_heuristic` and
  `random_legal` policies can be used
- exposes a matching CLI entrypoint through `scripts/run_live_experiment.py`

### What It Does Not Do Yet

- no end-to-end RL optimizer or trainer loop
- no simulator-first training loop
- no model registry beyond lightweight checkpoint helpers
- no claim that end-to-end agent training exists today

### Masked-DQN Baseline Components

The training package now includes the first reusable learner-core pieces for a
masked DQN baseline:

- `ReplayBuffer`: bounded in-memory storage for learner transitions with random
  tensor batch sampling over `(state, action_index, reward, next_state, done, mask)`
- `MaskedDQN`: a small PyTorch MLP that consumes the frozen 93-feature learner
  vector and emits 61 Q-values aligned to the frozen learner action indices
- mask-aware action selection helpers that only choose legal actions and fall
  back safely to `END_TURN` when a mask is malformed or degenerate
- lightweight checkpoint save/load helpers for model weights plus metadata

PyTorch is the first ML-side dependency in the repo because these baseline
components need a real tensor and module implementation for the trainer layer.

### Intended Extension Path

- learned policies can plug in through the same `Policy` contract or a
  `PolicyProvider` that resolves policies from checkpoints/configs
- future training code should keep using `ExperimentRunner` or its artifact
  conventions instead of adding a separate execution path
- dataset processing can build directly on `trajectory.jsonl` and
  `episodes.jsonl` without bypassing live rollout/replay contracts

### Artifact Structure

Each run writes:

- `config.json`: the validated experiment spec
- `metadata.json`: run timing, policy, fingerprint, and environment metadata
- `summary.json`: aggregate counts and rollout-level summary metrics
- `episodes.jsonl`: one line per episode result
- `trajectory.jsonl`: one line per replay entry, keyed by `episode_index`

### Example Config

See `configs/experiments/random_legal_collection.json`.

### Example Command

```bash
python scripts/run_live_experiment.py \
  --transport your_bridge_module:build_transport \
  --config configs/experiments/random_legal_collection.json
```
