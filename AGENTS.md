# AGENTS.md

## Mission
Build a Slay the Spire RL research stack, starting with a combat-only environment.

## Rules
- Never push directly to main.
- Prefer small PRs tied to one GitHub issue.
- Every behavior change must include or update tests.
- If you change interfaces or workflow, update docs/.
- Do not add major dependencies unless justified in the PR summary.

## Code standards
- Python 3.11+
- Use type hints for public functions
- Prefer dataclasses for state containers
- Keep modules small and composable
- Avoid hidden global state
- Favor deterministic, testable logic

## Required checks before opening a PR
- Run format
- Run lint
- Run tests
- Write a short summary of changes and risks

## Commands
- install: pip install -r requirements.txt
- test: pytest -q
- lint: ruff check .
- format: ruff format .

## Priorities
1. Correctness
2. Reproducibility
3. Testability
4. Simplicity
5. Performance

