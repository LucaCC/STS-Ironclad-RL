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
6. A future `ObservationEncoder`, `Policy`, `ActionContract`, and
   `RolloutRunner` layer consumes bridge outputs and emits legal actions
   through the same bridge interfaces.

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

## Next Layer Above The Bridge

The bridge should remain the lowest Python-facing boundary. The next layer in
this repo should add:

- an `ObservationEncoder` that turns `GameStateSnapshot` into stable
  policy-facing inputs
- an `ActionContract` that validates legal actions and maps them to
  `ActionCommand`
- a `RolloutRunner` that owns the live control loop
- structured replay logging that supplements the raw protocol trace

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
- trace logging interfaces
- tests that validate repo-side structure and serialization

This phase does not implement a full training stack or commit the repo to Gym
or Gymnasium as the core abstraction.

## Assumptions And Unknowns

- The exact transport used by CommunicationMod may be sockets, stdin or stdout,
  files, or another mod-exposed mechanism. The repo should keep that boundary
  abstract until a real host integration is tested.
- The exact schema exported by the game-side mod stack is not yet verified on a
  local machine. Protocol fields in this repo should therefore stay conservative
  and extensible.
- Some game actions may need additional context or IDs beyond simple action
  names.
- The boundary between full run-level state and combat-only state still needs to
  be confirmed during real integration.

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
