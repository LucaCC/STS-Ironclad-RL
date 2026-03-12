# Slay the Spire RL

Reinforcement-learning research stack for Slay the Spire, starting with a combat-only environment.

## Milestones
- Milestone 0: repo, agents, CI, automation
- Milestone 1: combat environment
- Milestone 2: baseline bots
- Milestone 3: evaluation suite
- Milestone 4: learning pipeline

## Current Focus

- Active milestone tracker: [docs/current_milestone.md](docs/current_milestone.md)
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
