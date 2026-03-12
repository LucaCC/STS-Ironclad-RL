## Architecture

This document outlines the intended high-level architecture for the project.

### Top-Level Layout

- `src/sts_ironclad_rl/env/`: Deterministic environment wrappers and interfaces for
  fast local RL iteration.
- `src/sts_ironclad_rl/integration/`: Live-game bridge contracts, session
  lifecycle, and trajectory logging for real-game validation.
- `src/sts_ironclad_rl/agents/`: Policy and value function implementations.
- `src/sts_ironclad_rl/training/`: Training loops, replay buffers, and
  optimization logic.
- `src/sts_ironclad_rl/evaluation/`: Evaluation harnesses and reporting helpers.
- `src/sts_ironclad_rl/utils/`: Shared utilities (logging, seeding, metrics,
  etc.).

### Current Environment Foundation

- `src/sts_ironclad_rl/env/state.py`: Immutable combat state primitives and seed-driven setup helpers.
- `src/sts_ironclad_rl/integration/protocol.py`: Typed bridge message contracts
  for host-provided game state, action requests, and trajectory records.
- `src/sts_ironclad_rl/integration/bridge.py`: Python-side bridge lifecycle
  around an injected transport implementation.
- `src/sts_ironclad_rl/integration/logger.py`: JSONL trace logging for later
  debugging, evaluation, and simulator cross-checks.
- Deterministic setup should flow through explicit seed arguments rather than hidden global RNG state.
- State transitions should be expressed as pure functions that return new state objects where practical.

### Key Design Principles

- **Modularity**: Environments, agents, and training loops should be loosely coupled.
- **Configurability**: Major experimental choices should be controlled through configuration files in `configs/`.
- **Testability**: Core logic should have unit tests in `tests/`.

This file should be updated as the implementation evolves.
