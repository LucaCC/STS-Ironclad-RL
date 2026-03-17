## Architecture

This document outlines the intended high-level architecture for the project.

## Milestone 1 Target

Milestone 1 is centered on a deterministic single-combat substrate for RL
training. The main implementation target is a local environment with explicit
seeding, testable combat state transitions, and a stepping interface suitable
for agent training loops.

### Top-Level Layout

- `src/env/`: Environment wrappers and interfaces for *Slay the Spire*.
- `src/agents/`: Policy and value function implementations.
- `src/training/`: Training loops, replay buffers, and optimization logic.
- `src/utils/`: Shared utilities (logging, seeding, metrics, etc.).

### Current Environment Foundation

- `src/sts_ironclad_rl/env/state.py`: Immutable combat state primitives and seed-driven setup helpers.
- `src/sts_ironclad_rl/env/combat.py`: Deterministic combat transition core for the milestone 1 action set.
- `src/sts_ironclad_rl/env/encoding.py`: Stable action ordering and flat observation encodings for training.
- `src/sts_ironclad_rl/env/training.py`: Thin wrapper that exposes the combat core through `reset` and `step`.
- `src/sts_ironclad_rl/training/`: Lightweight rollout and trainer scaffolding
  for seeded baseline episodes without committing to an RL library yet.
- Deterministic setup should flow through explicit seed arguments rather than hidden global RNG state.
- State transitions should be expressed as pure functions that return new state objects where practical.

## Track Separation

- Local deterministic environment: the primary training path and the default
  substrate for RL experimentation.
- Live-game bridge via CommunicationMod: a secondary validation and integration
  path used to compare assumptions against the real game and maintain smoke
  coverage.

### Key Design Principles

- **Modularity**: Environments, agents, and training loops should be loosely coupled.
- **Configurability**: Major experimental choices should be controlled through configuration files in `configs/`.
- **Testability**: Core logic should have unit tests in `tests/`.

This file should be updated as the implementation evolves.
