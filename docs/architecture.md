## Architecture

This document outlines the intended high-level architecture for the project.

### Top-Level Layout

- `src/sts_ironclad_rl/env/`: Deterministic combat environment state, actions,
  and transition rules.
- `src/sts_ironclad_rl/agents/`: Policy and value function implementations.
- `src/sts_ironclad_rl/training/`: Training loops, replay buffers, and
  optimization logic.
- `src/sts_ironclad_rl/evaluation/`: Evaluation harnesses and reporting helpers.
- `src/sts_ironclad_rl/utils/`: Shared utilities (logging, seeding, metrics,
  etc.).

### Current Environment Foundation

- `src/sts_ironclad_rl/env/state.py`: Immutable combat state primitives,
  deterministic card instances, monster intent cycles, and draw helpers.
- `src/sts_ironclad_rl/env/combat.py`: Structured combat actions, legal-action
  generation, and pure transition helpers for card play and end-turn
  resolution.
- Real-game smoke traces should inform naming and state shape, but the local
  battle environment should remain runnable without a live game connection.
- Deterministic setup should flow through explicit seed arguments rather than hidden global RNG state.
- State transitions should be expressed as pure functions that return new state objects where practical.

### Key Design Principles

- **Modularity**: Environments, agents, and training loops should be loosely coupled.
- **Configurability**: Major experimental choices should be controlled through configuration files in `configs/`.
- **Testability**: Core logic should have unit tests in `tests/`.

This file should be updated as the implementation evolves.
