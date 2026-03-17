# Current Milestone

This document tracks the currently active milestone so contributors do not have
to rely on chat history or local agent memory.

## Active Focus

Milestone 1 - Integrated Live CommunicationMod Control Pipeline

## Goal

Ship one coherent bridge-backed control path that can drive, evaluate, and
collect episodes from a real Slay the Spire process.

## Milestone 1 Target

Milestone 1 now centers one live-game-first execution path built around
CommunicationMod or an equivalent bridge:

- connect to a real game session
- request and validate state snapshots
- encode observations for a policy-facing contract
- map chosen actions back into bridge commands
- run the same rollout loop for smoke control, evaluation, and collection
- log raw traces and structured replay data for later analysis

## Tasks

- keep the integrated rollout/evaluation/collection path stable while real-game
  validation starts
- implement a concrete live-game transport adapter
- capture real bridge traces and refine protocol fields against observed data
- add the first learning-agent branch on top of the current rollout contracts
- keep simulator code limited to support functions, tests, or offline analysis

## Success Criteria

The repo exposes one bridge-backed observation -> policy -> action -> replay
pipeline, plus CLI entrypoints for evaluation and collection, with tests and
docs aligned around that path.
