## Agents in This Repository

This repository is intended to be used with AI coding assistants (like Cursor Agents) to streamline development. This document specifies how agents should behave when working here.

### General Guidelines

- Prefer small, focused changes that keep the repository in a working state.
- When adding new modules, also add or update tests in `tests/`.
- Keep documentation up to date whenever you introduce non-trivial behavior changes.

### File and Directory Conventions

- `src/`: Core source code. Organize by domain (e.g., `env/`, `agents/`, `training/`).
- `configs/`: Reusable experiment and training configurations.
- `scripts/`: One-off or reusable command-line entry points.
- `docs/`: Long-form documentation and technical notes.

### Git and CI

- Do not force-push to `main` unless explicitly instructed.
- Ensure CI (see `.github/workflows/ci.yml`) is green before merging feature branches into `main`.

