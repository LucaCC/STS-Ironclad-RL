# Live Game Bridge

## Purpose

The live-game bridge connects this repository to a locally installed copy of
*Slay the Spire* running through the mod stack. It is now the primary
implementation path for the next milestone. It exists to:

- prove the end-to-end control loop against the real game
- collect trajectories for future evaluation and training
- establish the policy-facing observation and action contracts
- provide the rollout path used by evaluation and later training

## System Components

### Local host machine

These components are outside this repository and must be installed and operated
locally by the user:

- *Slay the Spire*
- ModTheSpire
- BaseMod
- CommunicationMod, or an equivalent communication layer that can expose state
  and accept actions

### Repository responsibilities

This repository is responsible for:

- bridge architecture on the Python side
- protocol and message contract design
- session lifecycle scaffolding
- trajectory logging interfaces
- tests for repo-side integration code
- future debugging once protocol traces and logs are available
- the live-loop interfaces that sit above the bridge

### User responsibilities

The user is responsible for:

- installing *Slay the Spire* locally
- installing or placing ModTheSpire and required mods locally
- verifying the game launches successfully with mods
- handling Steam paths, local filesystem layout, and host-specific launch
  details

## Expected Control And Data Flow

1. The user launches *Slay the Spire* locally with the required mods.
2. The communication layer publishes structured game or combat state.
3. `src/sts_ironclad_rl/integration/bridge.py` opens a session and exchanges
   typed messages with that communication layer.
4. `src/sts_ironclad_rl/integration/protocol.py` defines the shape of state
   snapshots, action requests, and trajectory records.
5. `src/sts_ironclad_rl/integration/logger.py` writes traces for replay,
   debugging, and later training or evaluation workflows.
6. `src/sts_ironclad_rl/live/observation.py` encodes snapshots for policies.
7. `src/sts_ironclad_rl/live/actions.py` validates legal actions and maps them
   into bridge commands.
8. `src/sts_ironclad_rl/live/rollout.py` drives the shared rollout loop used by
   evaluation and collection entrypoints.

Data flow:

`game -> CommunicationMod/equivalent -> Python bridge -> observation encoder -> policy -> action contract -> command -> replay + trace logging`

## Component Roles

### ModTheSpire

Provides the mod loader that starts the instrumented game.

### BaseMod

Supplies common hooks and modding infrastructure used by many Slay the Spire
mods.

### CommunicationMod or equivalent

Acts as the communication boundary between the live game and external tooling.
This document assumes it can expose structured state and receive action
commands. Exact transport details are intentionally treated as configurable
rather than hard-coded facts.

### Python bridge code

Owns session lifecycle, validates message structure, and isolates transport
details from the rest of the repo.

### Logger

Records observation and action traces in a stable format so that later work can
debug bridge behavior, build datasets, and compare simulator output to the live
game.

## Initial Smoke Test

The first smoke test target is intentionally small:

1. launch the game locally with the required mods
2. connect from Python
3. read a combat state snapshot
4. send one legal action
5. log the before and after trace

Success here proves the bridge can observe the game, issue a command, and
record the resulting transition.

## Layer Above The Bridge

The bridge remains the lowest Python-facing boundary. The layer above it now
contains:

- an `ObservationEncoder` that turns `GameStateSnapshot` into stable
  policy-facing inputs
- an `ActionContract` that validates legal actions and maps them to
  `ActionCommand`
- a concrete `LiveEpisodeRunner` that implements the `RolloutRunner` contract
- structured replay serialization that supplements the raw protocol trace
- evaluation and collection entrypoints that reuse the same rollout path

## Local Setup Responsibilities

This repository does not claim to finish host-specific setup. Local setup work
still includes:

- locating the game install
- placing mod jars in the correct directories
- configuring any communication-mod settings
- confirming the game boots cleanly with mods enabled
- resolving operating-system-specific launch issues

Those steps depend on the user's machine and should be documented incrementally
once tested on real hosts.

## Repository Responsibilities

The repository-side deliverables for this phase are:

- a live-game bridge design document
- typed message contracts
- bridge scaffolding with explicit lifecycle boundaries
- live observation, action, rollout, replay, policy, and evaluation modules
- tests that validate repo-side structure and serialization

This phase does not implement a full training stack or commit the repo to Gym
or Gymnasium as the core abstraction.

## Concrete CommunicationMod Bridge

The current production path uses a small helper process launched by
CommunicationMod itself. The helper:

- prints `ready` so CommunicationMod keeps the child process alive
- reads JSON payloads from stdin
- translates them into the repo's `GameStateSnapshot` bridge payload
- exposes a local TCP socket for repo-side clients
- writes either a queued live action or a safe `STATE` poll command to stdout

This keeps the repo-side training and evaluation scripts on the existing
`BridgeTransport` abstraction while matching the proven CommunicationMod
child-process protocol.

### CommunicationMod config

Set CommunicationMod's `command=` entry to the helper process:

```text
command=python /Users/lucacc/Desktop/STS/STS-Ironclad-RL/scripts/communication_mod_bridge_helper.py --host 127.0.0.1 --port 8080
```

If you choose a different port, pass the same `--host` and `--port` values to
the repo scripts.

### Repo-side transport factory

Use the concrete socket transport factory:

```text
sts_ironclad_rl.integration.communication_mod:build_transport
```

### Console commands

Benchmark:

```bash
python scripts/run_live_benchmark.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --host 127.0.0.1 \
  --port 8080 \
  --config configs/benchmarks/baseline_eval.json
```

Training:

```bash
python scripts/train_live_dqn.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --host 127.0.0.1 \
  --port 8080 \
  --config configs/training/masked_dqn_baseline.json
```

Evaluation:

```bash
python scripts/evaluate_live_policy.py \
  --transport sts_ironclad_rl.integration.communication_mod:build_transport \
  --host 127.0.0.1 \
  --port 8080 \
  --policy simple_heuristic \
  --episodes 3
```

### Host And Port Behavior

- the helper binds the requested local host and port and waits for a repo-side
  TCP client
- the repo-side `SocketBridgeTransport` connects to that listener using the
  existing `BridgeConfig`
- one helper should be paired with one local game instance
- one live script should be connected to a helper at a time

### Known Limitations

- the helper currently translates only the action set used by the existing live
  rollout path: `play`, `end`, `choose`, `proceed`, and `leave`
- helper-side idle polling is implemented with `STATE`, which keeps the
  CommunicationMod request/response loop alive but may produce extra traffic
- the bridge assumes the current observed CommunicationMod payload shape,
  especially `available_commands`, `ready_for_command`, and nested `game_state`
- disconnect handling is intentionally minimal: timeout or socket loss bubbles
  up to the existing rollout interruption path

## Assumptions And Unknowns

- Some game actions may need additional context or IDs beyond simple action
  names.
- The boundary between full run-level state and combat-only state still needs to
  be confirmed during broader real-game validation.

## Risks

- CommunicationMod or an equivalent layer may not expose the exact hooks needed
  without additional mod work.
- Live-game timing and synchronization can introduce edge cases that the
  simulator does not have.
- Host-specific installation issues will block validation until the local setup
  works.
- If protocol traces are underspecified, later debugging and dataset generation
  will be expensive.

## Next Implementation Steps

- confirm the transport exposed by the local mod stack
- capture one or more real protocol traces from a host machine
- refine the protocol schema around observed fields instead of guesses
- implement a concrete transport adapter in `bridge.py`
- add a smoke-test script that connects, prints state, sends one action, and
  writes a trajectory log
