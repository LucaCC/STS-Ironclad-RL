# Slay the Spire RL

Reinforcement-learning research stack for Slay the Spire, centered on a
live-game-first Ironclad combat loop driven through CommunicationMod.

The current milestone is the first trainable agent path:

`live game -> rollout -> learner contract -> replay -> masked DQN training -> checkpoint -> evaluation -> benchmark comparison`

## Current Status

- one shared live rollout path for collection, evaluation, and training
- frozen learner contract with a 93-dim observation vector and 61-action space
- replay-backed PyTorch masked-DQN baseline with default hidden sizes `(128, 128)`
- benchmark workflow shared across `RandomLegalPolicy`, `SimpleHeuristicPolicy`,
  and checkpoint-backed `MaskedDQNPolicy`
- reproducible artifact layouts for collection runs, training runs, and
  benchmark comparisons

The repo does not yet ship a concrete CommunicationMod transport, a
simulator-first training path, or Rainbow-style DQN upgrades.

## Recommended Workflow

Install and validate the repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ruff format .
ruff check .
pytest -q
```

Run the first baseline benchmark:

```bash
python scripts/run_live_benchmark.py \
  --transport your_bridge_module:build_transport \
  --config configs/benchmarks/baseline_eval.json
```

Train the first masked-DQN baseline:

```bash
python scripts/train_live_dqn.py \
  --transport your_bridge_module:build_transport \
  --config configs/training/masked_dqn_baseline.json \
  --output-dir artifacts/training/masked_dqn_baseline
```

Evaluate the trained checkpoint directly:

```bash
python scripts/evaluate_live_policy.py \
  --transport your_bridge_module:build_transport \
  --policy dqn_checkpoint:artifacts/training/masked_dqn_baseline/checkpoints/checkpoint_final.pt \
  --policy-name masked_dqn \
  --episodes 3
```

Compare the checkpoint against the baselines:

```bash
python scripts/run_live_benchmark.py \
  --transport your_bridge_module:build_transport \
  --config configs/benchmarks/masked_dqn_vs_baselines.json
```

## Key Docs

- [docs/current_milestone.md](docs/current_milestone.md)
- [docs/trainable_agent_milestone.md](docs/trainable_agent_milestone.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/live_game_first_milestone.md](docs/live_game_first_milestone.md)
- [docs/learner_contract.md](docs/learner_contract.md)
- [docs/live_benchmarks.md](docs/live_benchmarks.md)

## Repo Layout

- `src/sts_ironclad_rl/integration/`: bridge protocol and transport boundary
- `src/sts_ironclad_rl/live/`: live observation, action, policy, rollout, and evaluation helpers
- `src/sts_ironclad_rl/training/`: learner contract, DQN baseline, artifacts, specs, and benchmark helpers
- `configs/`: canonical collection, training, and benchmark configs
- `scripts/`: thin live entrypoints built on package helpers
- `tests/`: deterministic unit, integration, and smoke coverage

## Limitations

- live execution is slow and noisy, so benchmark episode counts stay small
- the masked DQN baseline is intentionally minimal and not sample-efficient
- benchmark results are useful for integration validation and directional
  comparisons, not strong research claims
- future algorithm upgrades should extend the current learner/trainer path, not
  replace it with a second stack
