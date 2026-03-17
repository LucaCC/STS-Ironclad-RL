## Architecture

This document outlines the current high-level architecture for the project.

## Current Direction

The active milestone is live-game-first through CommunicationMod. The main
implementation target is a bridge-backed control loop around the real game,
with narrow typed contracts for observations, actions, rollout logging, and
evaluation.

### Top-Level Layout

- `src/sts_ironclad_rl/integration/`: Bridge protocol, session lifecycle, and
  raw trajectory logging for CommunicationMod-facing integration.
- `src/sts_ironclad_rl/live/`: Live-game observation and action contracts plus
  rollout-facing interfaces that sit above the bridge.
- `src/sts_ironclad_rl/agents/`: Future policy implementations that target the
  live contracts.
- `src/sts_ironclad_rl/training/`: Future rollout and training code built on
  the same live control path.
- `src/sts_ironclad_rl/evaluation/`: Future evaluation harnesses built on the
  live rollout path.
- `src/sts_ironclad_rl/utils/`: Shared utilities (logging, seeding, metrics,
  etc.).

### Current Live Foundation

- `src/sts_ironclad_rl/integration/protocol.py`: Typed bridge message contracts
  for host-provided game state, action requests, and trajectory records.
- `src/sts_ironclad_rl/integration/bridge.py`: Python-side bridge lifecycle
  around an injected transport implementation.
- `src/sts_ironclad_rl/integration/logger.py`: JSONL trace logging for later
  debugging, evaluation, and simulator cross-checks.
- `src/sts_ironclad_rl/live/contracts.py`: Policy-facing observation, action,
  replay, and rollout interfaces.
- `src/sts_ironclad_rl/live/actions.py`: Canonical live actions, legality
  helpers, and bridge-command mapping.
- `src/sts_ironclad_rl/live/replay.py`: Structured replay logging that keeps
  encoded observations and mapped commands easy to inspect without replacing the
  raw bridge trace.
- `src/sts_ironclad_rl/live/observation.py`: Typed observation parsing plus a
  stable flat/vector encoding layer built from bridge snapshots.
- Deterministic local environment code may remain for narrow testing support,
  but it is not the public control interface for this milestone.

Observation contents and invariants are documented in
`docs/live_observation_contract.md`.

## Track Separation

- Live-game bridge via CommunicationMod: the primary implementation path.
- Local deterministic environment: secondary support code for narrow tests and
  offline analysis only.

### Key Design Principles

- **Modularity**: Environments, agents, and training loops should be loosely coupled.
- **Configurability**: Major experimental choices should be controlled through configuration files in `configs/`.
- **Testability**: Core logic should have unit tests in `tests/`.

This file should be updated as the implementation evolves.
