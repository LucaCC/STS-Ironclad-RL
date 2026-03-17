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

## PR 19 Recommendation

### Keep

- the higher bar for explicit action contracts and legal-action generation as a
  design lesson
- the emphasis on deterministic, testable state transitions where local logic
  still exists

### Discard

- the card-play combat environment redesign as the primary platform
- removal of `src/sts_ironclad_rl/integration/`
- deletion of bridge smoke and protocol coverage
- milestone framing that centers simulator expansion

### Re-scope

- any action-shape work should move into the bridge-facing `ActionContract`
  layer, where policy actions map to real game commands instead of simulator
  transitions

## PR 20 Recommendation

### Keep

- the idea of a narrow `Policy` interface
- the separation between observation encoding, rollout execution, and
  evaluation/reporting
- the instinct to share one rollout path between evaluation and data
  collection

### Discard

- `CombatTrainingEnv` as the central abstraction
- fixed-index simulator action encoding as the public contract
- baseline trainer and evaluator code that depends on deterministic combat
  state instead of live bridge snapshots
- docs that describe simulator-first Milestone 1 scope
- removal of `src/sts_ironclad_rl/integration/`

### Re-scope

- `encoding.py` becomes a live-game `ObservationEncoder`
- `training/rollout.py` becomes a bridge-backed `RolloutRunner`
- `evaluation/evaluator.py` survives only if rebuilt on top of the live rollout
  path
- baseline policies should target encoded live observations plus the bridge
  action contract, not `CombatState`

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

Non-goals:

- no commitment yet to tensor libraries or model-specific batching
- no simulator-only feature schema as the repo-wide public contract

### `ActionContract`

Responsibilities:

- define the policy-facing action namespace
- validate that a chosen action is legal for the current snapshot
- map a chosen action into `ActionCommand`

Design bias:

- keep the first version close to the bridge payloads
- prefer explicit string or structured action IDs over fixed simulator indices

### `RolloutRunner`

Responsibilities:

- execute one live episode or bounded session slice
- drive the control loop around bridge I/O
- emit structured replay records
- serve as the single execution path for both evaluation and data collection

Non-goals:

- no training algorithm ownership
- no Gym compatibility layer as the primary API

### Replay / Structured Logging

Use two layers:

- raw protocol logging for exact bridge traces
- structured replay entries for encoded observations, chosen actions, rewards,
  and terminal metadata

This separation keeps debugging grounded in the real wire contract while still
supporting policy and offline-analysis workflows.

### `Policy`

The repo-wide policy contract should stay minimal:

- input: encoded observation plus legal actions
- output: one policy action decision

Policies should not depend on simulator internals.

### Evaluation Harness

Build evaluation on top of `RolloutRunner`, not beside it.

First harness scope:

- repeated live smoke episodes when the game is available
- deterministic replay-file evaluation when live control is unavailable
- simple PM-facing metrics such as completion rate, action validity, step
  counts, and terminal outcomes

## Parallel Workstreams

These can proceed in parallel once the shared contracts stay stable.

### Workstream A: Transport And Real Trace Capture

- implement one concrete transport for the CommunicationMod setup in use
- capture and check in redacted example traces
- tighten protocol types around observed fields

### Workstream B: Observation And Action Contracts

- implement the first real `ObservationEncoder`
- define the first `ActionContract` against observed legal actions
- document unsupported game states explicitly

### Workstream C: Rollout And Replay

- build the bridge-facing `RolloutRunner`
- add structured replay writing and replay readers
- make raw bridge logs and structured logs correlate by session and step index

### Workstream D: Policies And Evaluation

- add one no-op or rule-based live policy
- build an evaluation harness on top of the rollout path
- define success metrics for smoke runs and data-collection runs

## Merge And Conflict Risks

- PR 19 and PR 20 both overlap heavily with `README.md`, `docs/architecture.md`,
  `docs/current_milestone.md`, `src/sts_ironclad_rl/__init__.py`,
  `src/sts_ironclad_rl/env/`, and tests.
- Both PRs also delete the current `integration/` package, which directly
  conflicts with the new project direction.
- Merging either branch wholesale would require follow-up revert work and would
  muddy the package contract for the next threads.

## Next Threads Checklist

- preserve `src/sts_ironclad_rl/integration/` as the bridge root
- implement a concrete transport adapter
- capture one real game trace and lock the protocol against it
- build the first bridge-backed `ObservationEncoder`
- build the first bridge-backed `ActionContract`
- build `RolloutRunner` on top of the live bridge
- add structured replay logging alongside raw protocol logs
- add a minimal rule-based policy against encoded live observations
- add evaluation helpers that consume the rollout path
- keep simulator changes narrowly scoped to support tests or offline analysis,
  not as the primary training substrate
