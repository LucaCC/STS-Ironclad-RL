# Live Replay Format

Structured replay files capture the bridge-backed control loop in an
inspectable format that is useful for debugging first and later offline
analysis second.

## Purpose

The live-game path keeps two logging layers:

- `src/sts_ironclad_rl/integration/logger.py` stores raw protocol-oriented
  traces tied closely to bridge traffic.
- `src/sts_ironclad_rl/live/replay.py` stores structured rollout records for
  encoded observations, legal actions, chosen actions, mapped commands, rewards,
  and terminal or failure metadata.

Use the raw trace to debug transport and wire-contract issues. Use structured
replay files to inspect rollout behavior and policy decisions step by step.

## Format

- file format: JSON Lines (`.jsonl`)
- one line per rollout step
- each line includes `schema_version`

Each replay line contains:

- `session_id`
- `step_index`
- `recorded_at`
- `state_reference`: optional compact pointer back to a raw trace or external
  snapshot artifact
- `observation`: encoded observation payload, including the source
  `GameStateSnapshot`
- `legal_action_ids`
- `action`: the policy-facing action decision, if present
- `command`: the mapped `ActionCommand` sent to the bridge, if present
- `reward`
- `terminal`
- `outcome`
- `failure`: interruption or error metadata
- `metadata`: free-form run annotations

## Intended Use

- debugging first: inspect legality, encoded state, and bridge-command mapping
  line by line
- future training support: convert replay files into later analysis or dataset
  jobs without introducing a heavy storage layer now

## Current Limitations

- the schema is intentionally narrow and does not define manifests, sharding, or
  compression
- `observation.snapshot` currently embeds the full bridge snapshot; callers that
  want smaller files should use `state_reference` and trim `raw_state` upstream
