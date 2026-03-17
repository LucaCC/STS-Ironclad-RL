# Current Milestone

This document tracks the currently active milestone so contributors do not have
to rely on chat history or local agent memory.

## Active Focus

Milestone 1 - Deterministic Single-Combat Environment

## Goal

Deliver a trainable deterministic combat substrate for Ironclad RL experiments.

## Milestone 1 Target

Milestone 1 targets a local, deterministic, single-combat environment that can
be stepped, seeded, tested, and used as the primary training substrate.

## Track Separation

- Local deterministic environment: primary path for RL training, reproducible
  experiments, and environment iteration.
- Live-game bridge: secondary path for validation, smoke testing, and
  integration against a real Slay the Spire process via CommunicationMod.

## Tasks

- complete deterministic combat state and transition foundations
- expose a trainable combat stepping interface with legal action handling
- add a minimal seeded rollout and baseline trainer scaffold for end-to-end
  milestone 1 runs, with evaluation reusing the same wrapper-level rollout path
- keep smoke and integration hooks sufficient to validate assumptions against
  the live game

## Success Criteria

A local seeded combat scenario can be stepped deterministically and supports the
action/state interfaces needed for training-oriented experiments, including a
small baseline rollout path that logs core combat metrics.
