# Live-Game-First Milestone Plan

This document is the source of truth for the next implementation wave.

## Decision

Prioritize the live CommunicationMod connection to the real game over further
deterministic simulator expansion.

## Why

- `main` already contains a usable bridge scaffold with typed protocol,
  session lifecycle validation, and JSONL trace logging.
- PR 19 and PR 20 both invest heavily in simulator-first abstractions.
- The project no longer wants Gym or Gymnasium as the core loop.
- The next bottleneck is a reliable bridge-facing RL/data-collection loop, not
  additional local combat mechanics.

## Repository Recommendation

Merge neither PR 19 nor PR 20 as-is.

Instead:

- keep `main` as the base because it preserves the live bridge package
- keep selected concepts from PR 20, but re-scope them onto live-game
  interfaces instead of deterministic combat wrappers
- discard the simulator-first rewrites from PR 19

## Reconciled Architecture

### `LiveGameSession`

Use the existing bridge session and transport boundary on `main` as the system
edge:

- `LiveGameBridge` owns connect, request-state, send-action, and receive-state
- higher layers should not talk directly to transport details
- the next step is a bridge-facing control loop that sequences
  observe -> encode -> choose -> map -> command -> log

### `ObservationEncoder`

Responsibilities:

- transform `GameStateSnapshot` into a stable policy input object
- expose legal actions in policy-facing form
- preserve enough metadata for replay and debugging

### `ActionContract`

Responsibilities:

- define the policy-facing action namespace
- validate that a chosen action is legal for the current snapshot
- map a chosen action into `ActionCommand`

### `RolloutRunner`

Responsibilities:

- execute one live episode or bounded session slice
- drive the control loop around bridge I/O
- emit structured replay records
- serve as the single execution path for both evaluation and data collection

## Parallel Workstreams

- Transport and real trace capture
- Observation and action contracts
- Rollout and replay
- Policies and evaluation
