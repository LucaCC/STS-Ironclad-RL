# Current Milestone

This document tracks the currently active milestone so contributors do not have
to rely on chat history or local agent memory.

## Active Focus

Milestone 1 - Live CommunicationMod Bridge And Data Collection

## Goal

Deliver the first bridge-backed RL control loop and logging substrate against a
real Slay the Spire process.

## Milestone 1 Target

Milestone 1 targets a live-game-first control loop built around
CommunicationMod or an equivalent bridge:

- connect to a real game session
- request and validate state snapshots
- encode observations for a policy-facing contract
- map chosen actions back into bridge commands
- log raw traces and structured replay data for later analysis

## Tasks

- implement a concrete live-game transport adapter
- capture real bridge traces and refine protocol fields against observed data
- add the first `ObservationEncoder` and `ActionContract`
- add a bridge-backed `RolloutRunner` for smoke runs and data collection
- preserve structured logging suitable for replay and evaluation
- keep simulator code limited to support functions, tests, or offline analysis

## Success Criteria

A live session can be connected, observed, stepped through a policy-facing
action contract, and logged in a form suitable for replay, debugging, and
future training experiments.
