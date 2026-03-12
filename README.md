# Slay the Spire RL

Reinforcement-learning research stack for Slay the Spire, starting with a combat-only environment.

The primary training path is a deterministic local battle environment. A
CommunicationMod-based live-game bridge is secondary and exists for validation,
smoke testing, and integration against the real game.

## Milestones
- Milestone 0: repo, agents, CI, automation
- Milestone 1: combat environment
- Milestone 2: baseline bots
- Milestone 3: evaluation suite
- Milestone 4: learning pipeline

## Current Focus

- Active milestone tracker: [docs/current_milestone.md](docs/current_milestone.md)
- Milestone 1 target: a trainable deterministic single-combat substrate
- Live-game bridge design: [docs/live_game_bridge.md](docs/live_game_bridge.md)

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

## Training Scaffold

Milestone 1 now includes a minimal baseline-oriented training scaffold under
`src/sts_ironclad_rl/training/`. It deliberately avoids a full RL library and
instead provides deterministic seeded rollouts, baseline policy evaluation, and
basic metric logging for the first combat slice.

Run it with:

```bash
python -m sts_ironclad_rl.training --train-policy heuristic --eval-policy random --train-episodes 10 --eval-episodes 5 --seed 7
```

The command emits per-episode JSON logs for the training phase and a final
evaluation summary with episode reward, win rate, combat length, remaining HP,
and HP delta.
