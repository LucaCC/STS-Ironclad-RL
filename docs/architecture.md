## Architecture

This document outlines the intended high-level architecture for the project.

### Top-Level Layout

- `src/env/`: Environment wrappers and interfaces for *Slay the Spire*.
- `src/agents/`: Policy and value function implementations.
- `src/training/`: Training loops, replay buffers, and optimization logic.
- `src/utils/`: Shared utilities (logging, seeding, metrics, etc.).

### Key Design Principles

- **Modularity**: Environments, agents, and training loops should be loosely coupled.
- **Configurability**: Major experimental choices should be controlled through configuration files in `configs/`.
- **Testability**: Core logic should have unit tests in `tests/`.

This file should be updated as the implementation evolves.

