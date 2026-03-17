# Live Action Contract

## Purpose

`src/sts_ironclad_rl/live/` defines the policy-facing live action namespace for
CommunicationMod-backed control.

The current contract is intentionally small and maps typed actions into
`ActionCommand(command=..., arguments=...)` objects without assuming a
simulator, Gym interface, or a finished rollout runner.

## Supported Actions

The current `CommunicationModActionContract` supports:

- `play_card:<hand_index>` for playable untargeted cards
- `play_card:<hand_index>:<monster_index>` for playable targeted cards
- `end_turn`
- `choose:<choice_index>`
- `proceed`
- `leave`

Policy-facing indices are zero-based. The bridge mapping keeps those stable and
converts only where CommunicationMod conventions differ. Today that means card
plays map to `command="play"` with a one-based `card_index`, while
`target_index` and `choice_index` remain zero-based in the command arguments.

## Validation Rules

The action contract validates against the current `GameStateSnapshot` before
emitting commands:

- command availability must be present in `snapshot.available_actions`
- card plays require combat context, a valid hand index, and a playable card
- targeted cards require a live monster target
- untargeted cards reject target arguments
- choice actions require a valid index into `raw_state["choice_list"]`

This keeps malformed action IDs and obviously unsafe commands out of the bridge
layer.

## Out Of Scope

Not supported in this wave:

- potion actions
- map-path or shop-item semantic wrappers beyond generic `choose:<index>`
- simulator-derived legality or combat reconstruction
- rollout execution, policy packages, or transport-specific serialization to raw
  socket/stdin command strings
