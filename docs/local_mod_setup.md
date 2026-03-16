# Local Slay the Spire Mod Setup and Validation

This runbook covers the human-owned local setup needed before the repository can
validate a live-game bridge against a real Slay the Spire process.

It is intentionally practical:

- it focuses on what must exist on the host machine
- it avoids hard-coding OS-specific paths unless called out as examples
- it does not define or implement the bridge itself

## Scope

Use this document when the goal is:

- launch Slay the Spire locally with the required mods enabled
- confirm the game starts cleanly under a mod loader
- gather enough evidence to debug launch or communication failures
- hand the setup back to repo-side bridge work once logs and traces exist

This document is not the source of truth for bridge protocol details, Python-side
transport code, or training environment design.

## Responsibilities

### User-local responsibilities

The local machine owner is responsible for:

- installing Slay the Spire locally
- installing and enabling the required mod loader and mods locally
- confirming the game reaches a usable launch state with mods enabled
- collecting logs, screenshots, and reproduction notes if the game or mod stack fails
- confirming whether a first communication smoke test appears to work from the host side

### Repo responsibilities

The repository is responsible for:

- bridge architecture and integration design
- Python-side integration code and protocol handling
- logging and trace formats used for later debugging
- follow-up debugging once host-side evidence is available

If the host machine cannot launch the game with the expected mods, the next step
is usually to fix the local environment first rather than changing repo code.

## Required Local Components

Prepare these components at a high level:

- `Slay the Spire`: the locally installed game
- `ModTheSpire`: the mod loader used to launch the game with mods
- `BaseMod`: common dependency used by many Slay the Spire mods
- `CommunicationMod` or an equivalent communication layer: the game-side mod that
  exposes state and actions to an external process

The exact installation path, packaging format, launcher behavior, and mod folder
layout may differ by operating system, storefront, and local setup. Treat any
path examples you find elsewhere as examples, not guaranteed facts.

## Prerequisites Checklist

Before trying any bridge validation:

- confirm the game is installed and can launch normally without the bridge workflow
- confirm you have the correct versions of `ModTheSpire`, `BaseMod`, and the
  communication-layer mod for your local game build
- confirm you know where your local mod loader and game logs are written
- confirm Java or other runtime requirements for the mod loader are satisfied on
  your machine
- confirm you can rerun the same launch flow repeatedly without changing multiple variables at once

## Install and Launch Verification

Use the smallest possible validation loop first.

1. Install or update `ModTheSpire`.
2. Install `BaseMod`.
3. Install `CommunicationMod` or the equivalent bridge-facing mod.
4. Launch the game through the mod loader with only the minimum required mods enabled if possible.
5. Confirm the game reaches the main menu or another stable in-game state without crashing.
6. Confirm the mod loader output does not show obvious missing dependency or version mismatch errors.

The first goal is not "full bridge success." The first goal is "the game starts
cleanly with the required mod stack."

## First Smoke-Test Target

The first human setup smoke test should be narrow and observable:

- the game launches with the communication-layer mod enabled
- the mod loader shows that required mods were detected and loaded
- the game reaches a stable screen without immediate crash or repeated error spam
- at least one host-side artifact exists that suggests the communication layer initialized

Examples of acceptable first artifacts:

- a launcher log showing the communication mod loaded
- a game log line indicating bridge-related initialization
- a console message or trace file produced by the communication layer

Do not expand the scope to full action execution until this basic smoke test is stable.

## Communication Validation Checklist

Once launch is stable, verify only the minimum bridge-facing behavior:

- confirm the communication layer appears to initialize during startup
- confirm any expected port, socket, file, or log-based endpoint is created if the chosen mod exposes one
- confirm the game can remain open long enough for an external process to attach or observe state
- record what "success" looks like on your machine in one or two sentences

If communication still fails, stop and capture evidence before changing repo code.

## Evidence to Collect When Something Fails

Capture enough evidence that repo-side debugging can reason about a real host state.

Record:

- operating system and game distribution source
- game version if visible
- versions of `ModTheSpire`, `BaseMod`, and the communication-layer mod
- exact launch method used
- whether the failure happened before launch, during mod loading, at the main menu, or during an attempted connection
- the smallest reproducible sequence of steps

Save:

- mod loader logs
- game logs
- console output
- screenshots of visible error dialogs
- any bridge-related trace or output file produced locally

If possible, note timestamps so logs from multiple components can be correlated.

## Repo Assumptions vs Host-Specific Facts

### Repo-side assumptions

The repository can reasonably assume that:

- a valid local Slay the Spire installation exists
- the required mod loader and dependencies are installed correctly
- the communication layer is present and enabled
- logs or traces can be shared back for debugging when failures occur

### Host-specific facts

The repository should not assume:

- exact install directories
- exact launch commands
- exact log file paths
- identical behavior across operating systems or storefront builds
- that a specific communication mod version works everywhere without validation

When documenting or reporting local setup, label machine-specific details clearly
as examples or local observations.

## Handoff Back to Repo Work

After the local setup is validated, hand off the following to repo-side work:

- whether the game launches cleanly with the required mods
- whether the communication layer appears to initialize
- the logs and traces captured from the smoke test
- any observed mismatch between expected and actual communication behavior

That evidence is the input for future bridge debugging, protocol adjustments, and
Python-side integration validation.

## Practical Exit Criteria

This runbook has done its job when all of the following are true:

- the local machine can launch Slay the Spire with the required mods enabled
- there is at least one concrete artifact showing the communication layer loaded
- failures, if any, have supporting logs or screenshots attached
- the next blocker is clearly repo-side integration work rather than unknown local setup state
