# Slay the Spire RL

Reinforcement-learning research stack for Slay the Spire, starting with a
live-game control loop for Ironclad combat via CommunicationMod.

The current implementation priority is the bridge to a real Slay the Spire
process. Deterministic simulator code on `main` remains useful for tests and
offline reasoning, but it is no longer the architectural center of the next
milestone.

## Milestones
- Milestone 0: repo, agents, CI, automation
- Milestone 1: live-game bridge and data collection foundation
- Milestone 2: live-game observation and action contracts
- Milestone 3: rollout, replay, and evaluation harness
- Milestone 4: first policy-training pipeline

## Current Focus

- Active milestone tracker: [docs/current_milestone.md](docs/current_milestone.md)
- Architecture and integration direction: [docs/architecture.md](docs/architecture.md)
- Live-game bridge plan: [docs/live_game_bridge.md](docs/live_game_bridge.md)
- PR reconciliation and next workstreams: [docs/live_game_first_milestone.md](docs/live_game_first_milestone.md)

## Project Structure

- `AGENTS.md`: Rules, code standards, and workflow expectations for contributors and AI agents.
- `docs/`: Project vision, architecture, and roadmap documents.
- `src/`: Source code for the RL environment, agents, and training loops.
- `tests/`: Unit and integration tests.
- `configs/`: Configuration files (hyperparameters, experiment setups, etc.).
- `scripts/`: Utility scripts for training, evaluation, and data processing.
- `tasks/`: Task templates and checklists.
- `.github/workflows/`: CI configuration.

## Developer setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

## Local workflow

Run the same checks locally before opening a PR:

```bash
pre-commit run --all-files
```

### Commands

```bash
ruff format .
ruff check .
pytest -q
```

Pre-commit runs:

- `ruff format --check .`
- `ruff check .`
- `pytest -q`

## Immediate Direction

Near-term implementation should converge on these repo-level concepts:

- `LiveGameBridge` and `LiveGameBridgeSession` as the bridge-facing session and
  control boundary
- `ObservationEncoder` for transforming bridge snapshots into policy inputs
- `ActionContract` for legal-action exposure and command mapping
- `RolloutRunner` for episode execution against the live game
- replay and structured logging for dataset generation, debugging, and audit
- a minimal `Policy` interface and evaluation harness built on the same rollout
  path

This repo intentionally does not adopt Gym or Gymnasium as the core abstraction
for that loop.
