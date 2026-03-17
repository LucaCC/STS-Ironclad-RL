# Live Observation Contract

## Purpose

`src/sts_ironclad_rl/live/observation.py` defines the first RL-facing
observation layer for live CommunicationMod snapshots.

The contract has two stages:

1. parse loose bridge payloads into typed observation dataclasses
2. export a stable flat feature map or numeric vector for policies

This keeps protocol parsing separate from policy-facing normalization.

## Included State

The typed `LiveObservation` includes:

- snapshot-level metadata:
  - schema version
  - `screen_state`
  - `screen_type`
  - `room_phase`
  - `action_phase`
  - `in_combat`
  - `floor`
  - `act`
  - `ascension_level`
  - `gold`
  - class name
  - `choice_list`
  - raw bridge `available_actions`
- combat metadata when `raw_state["combat_state"]` is present:
  - turn number
  - player HP, max HP, block, and energy
  - draw, discard, and exhaust pile sizes
  - normalized hand-card slots
  - normalized enemy slots in bridge order
  - enemy and targetable-enemy counts

The encoded observation also exposes policy-facing `legal_action_ids` derived
through the live `CommunicationModActionContract`.

## Invariants

The encoder guarantees:

- `schema_version` is fixed at `live_observation.v1` for this contract version
- `legal_action_ids` are derived from the action contract, not copied directly
  from bridge command names
- the flat dict export always emits the same keys for a given
  `ObservationLayout`
- the vector export always emits the same length and order for a given
  `ObservationLayout`
- missing or malformed optional fields degrade to explicit defaults instead of
  raising:
  - missing integers become `0` or `-1` depending on whether the field is a
    value or sentinel-coded scalar
  - missing booleans become `False`
  - missing strings become `None` in typed form and `""` in flat dict form
  - missing combat state produces a valid non-combat observation with padded
    zeroed combat slots
- enemy and hand slot ordering matches the incoming bridge ordering
- vector exports contain only numeric or boolean-derived values and require no
  third-party ML libraries

## Current Limits

- no vocabulary or embedding scheme is defined for card names, enemy names, or
  intent strings
- only the fields already observed in repo docs and CommunicationMod smoke
  tooling are normalized here
- the encoder pads to a fixed number of hand and enemy slots for policy-facing
  exports, so larger live states are truncated in the flat/vector form while
  remaining present in the typed `LiveObservation`

If real traces expose additional stable fields, extend the typed dataclasses
first and then deliberately version the contract if the flat/vector schema must
change.
