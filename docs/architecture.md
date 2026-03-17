## Architecture

This document outlines the intended high-level architecture for the project.

## Milestone 1 Target

Milestone 1 is centered on a live-game-first RL and data-collection substrate.
The main implementation target is a custom bridge-backed control loop, not a
Gym-style simulator wrapper.

### Top-Level Layout

- `src/sts_ironclad_rl/integration/`: Bridge protocol, transport boundary,
  session lifecycle, and raw trace logging.
- `src/sts_ironclad_rl/live/`: Bridge-facing observation, action, policy,
  rollout, and replay contracts.
- `src/sts_ironclad_rl/training/`: Lightweight experiment specs, collection
  runners, and artifact layout built on the shared live rollout interface.
- `src/sts_ironclad_rl/env/`: Local deterministic helpers kept narrow for tests,
  offline reasoning, and non-primary support workflows.
- future agent, training, and evaluation packages should build on the live
  contracts rather than define competing loop abstractions.

### Bridge Foundation

- `src/sts_ironclad_rl/integration/protocol.py`: Typed bridge message contracts
  for host-provided game state, action requests, and trajectory records.
- `src/sts_ironclad_rl/integration/bridge.py`: Python-side bridge lifecycle
  around an injected transport implementation.
- `src/sts_ironclad_rl/integration/logger.py`: JSONL trace logging for later
  debugging, evaluation, replay correlation, and simulator cross-checks.

### Live RL Contracts

- `LiveGameBridge` and `LiveGameBridgeSession` are the bridge-facing control
  boundary. Higher layers should work through these types rather than bypassing
  transport/session logic.
- `src/sts_ironclad_rl/live/`: Bridge-facing observation encoding, action
  mapping, baseline policies, rollout execution, replay utilities, and
  lightweight evaluation helpers for CommunicationMod-driven runs.
- `ObservationEncoder` translates `GameStateSnapshot` into stable policy-facing
  observations while preserving metadata needed for replay/debugging.
- `ActionContract` owns the policy-facing action namespace and maps chosen
  actions into `ActionCommand`.
- `RolloutRunner` executes the live control loop using the bridge, encoder,
  action contract, and policy.
- replay logging records encoded observations, actions, rewards, and terminal
  metadata without discarding the raw bridge trace.
- the evaluation harness should call into the same rollout path used for data
  collection.
- the training/experimentation scaffold should call the same rollout runner and
  write reproducible run artifacts rather than introducing a second loop.

### Non-Goals For This Milestone

- no Gym or Gymnasium core abstraction
- no new deterministic simulator as the main implementation path
- no training-library commitment before the live loop and replay contracts
  exist

## Track Separation

- Live-game bridge via CommunicationMod: the primary implementation path for
  the next phase.
- Local deterministic environment: a secondary support path for tests, offline
  experimentation, and targeted reasoning, not the repo-wide training contract.

### Key Design Principles

- **Modularity**: Environments, agents, and training loops should be loosely coupled.
- **Configurability**: Major experimental choices should be controlled through configuration files in `configs/`.
- **Testability**: Core logic should have unit tests in `tests/`.
- **Single execution path**: evaluation and data collection should share one
  rollout path over the live bridge.
- **Bridge fidelity first**: observed real-game fields should drive contract
  design ahead of simulator convenience.

This file should be updated as the implementation evolves.
