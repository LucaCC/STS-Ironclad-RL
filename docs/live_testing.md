# Live Pipeline Testing

This document defines the default test split for the live-game RL pipeline.

## Goals

- keep routine test runs fast and deterministic
- exercise the real repo-side observation, action, rollout, and evaluation code
- avoid depending on a running Slay the Spire instance for CI or local edit-test
  loops

## Test Layers

### Unit tests

Location: `tests/live/unit/`

Use these for pure or mostly pure repo-side logic:

- action ID parsing and command mapping
- observation normalization and feature extraction
- policy selection rules
- small contract helpers that do not need bridge I/O

Bias:

- prefer representative snapshot fixtures over mocks
- assert stable IDs, features, and validation behavior directly

### Contract tests

Location: `tests/live/contract/`

Use these for boundaries between repo components:

- observation encoder output consumed by action contracts or policies
- rollout runner fallback behavior when one layer omits optional data
- snapshot/action namespace expectations that future threads must preserve

Bias:

- test the seam, not just one function in isolation
- keep transports fake, but keep encoded observations and commands real

### Bridge-adjacent smoke tests

Location: `tests/live/smoke/`

Use these for fast end-to-end slices of the Python-side live loop:

- `LiveGameBridge` plus `LiveEpisodeRunner`
- replay collection
- evaluation summary generation
- baseline policy integration with the live observation and action contracts

Bias:

- use fake transports and scripted snapshots
- prove the observe -> encode -> choose -> map -> send path works
- keep the scenarios short enough for routine `pytest -q`

## Manual Verification Still Required

The following still need real-game validation through CommunicationMod and are
not routine test targets:

- transport wiring against the actual mod/runtime setup
- schema drift in real snapshots not covered by checked-in fixtures
- timing, polling cadence, and bridge disconnect behavior under live latency
- unsupported screens, modal flows, and multi-step interactions not yet encoded
- whether chosen commands are accepted by the live game exactly as expected

## Conventions For Future Threads

- add new live fixtures in `tests/live/factories.py` before copying snapshot
  dictionaries into test files
- add reusable bridge or policy doubles in `tests/live/fakes.py`
- default new coverage to unit tests, promote only true seams to contract tests,
  and reserve smoke tests for short end-to-end slices
- when adding unsupported live-game behavior, document whether it is covered by
  unit tests, smoke tests, or manual verification
