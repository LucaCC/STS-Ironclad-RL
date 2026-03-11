# Slay the Spire RL

Reinforcement-learning research stack for Slay the Spire, starting with a combat-only environment.

## Milestones
- Milestone 0: repo, agents, CI, automation
- Milestone 1: combat environment
- Milestone 2: baseline bots
- Milestone 3: evaluation suite
- Milestone 4: learning pipeline

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

### Commands

```bash
ruff format .
ruff check .
pytest -q
```

