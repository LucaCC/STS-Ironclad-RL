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

The repo now ships a minimal live CommunicationMod socket bridge. It still does
not ship a simulator-first training path or Rainbow-style DQN upgrades.

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
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --config configs/benchmarks/baseline_eval.json
```

Train the first masked-DQN baseline:

```bash
python scripts/train_live_dqn.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --config configs/training/masked_dqn_baseline.json \
  --output-dir artifacts/training/masked_dqn_baseline
```

Evaluate the trained checkpoint directly:

```bash
python scripts/evaluate_live_policy.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --policy dqn_checkpoint:artifacts/training/masked_dqn_baseline/checkpoints/checkpoint_final.pt \
  --policy-name masked_dqn \
  --episodes 3
```

Compare the checkpoint against the baselines:

```bash
python scripts/run_live_benchmark.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --config configs/benchmarks/masked_dqn_vs_baselines.json
```

## CommunicationMod Live Bridge

Configure CommunicationMod to launch the helper process through its `command=`
setting:

```text
command=python /Users/lucacc/Desktop/STS/STS-Ironclad-RL/scripts/communication_mod_bridge_helper.py --host 127.0.0.1 --port 8080
```

Then run the repo-side entrypoints from a separate terminal:

```bash
python scripts/run_live_benchmark.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --host 127.0.0.1 \
  --port 8080 \
  --config configs/benchmarks/baseline_eval.json
```

```bash
python scripts/train_live_dqn.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --host 127.0.0.1 \
  --port 8080 \
  --config configs/training/masked_dqn_baseline.json
```

```bash
python scripts/evaluate_live_policy.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --host 127.0.0.1 \
  --port 8080 \
  --policy simple_heuristic \
  --episodes 3
```

Host and port must match on both sides. The helper binds a local TCP listener
and translates between the repo's `BridgeTransport` envelopes and
CommunicationMod's launched child-process stdin/stdout protocol.

Current limitations:

- one local helper process per game instance and one active repo client per helper
- the helper uses `STATE` as its idle poll command, so live runs depend on that
  CommunicationMod command remaining available
- only the current action set is translated: `play`, `end`, `choose`,
  `proceed`, and `leave`
- unsupported screens or schema drift still require manual validation against
  the live game

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
