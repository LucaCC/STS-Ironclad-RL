## Live Training Scaffold

This scaffold turns the existing live rollout path into a lightweight
experimentation substrate for data collection.

### What It Does Today

- defines explicit `ExperimentSpec` configs for live collection runs
- runs repeated episodes through the existing `RolloutRunner` contract
- writes stable run artifacts under `artifacts/experiments/<experiment>/<run_id>/`
- saves replay-backed `trajectory.jsonl` and per-episode `episodes.jsonl`
- records `config.json`, `metadata.json`, and `summary.json` for reproducibility
- leaves policy selection open so current heuristic/random policies can be used

### What It Does Not Do Yet

- no RL optimizer, replay buffer sampler, or gradient-based learner
- no simulator-first training loop
- no checkpoint management or model registry
- no claim that end-to-end agent training exists today

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

See [configs/experiments/random_legal_collection.json](/Users/lucacc/Desktop/STS/STS-Ironclad-RL/configs/experiments/random_legal_collection.json).
