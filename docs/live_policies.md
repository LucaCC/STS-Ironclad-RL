# Live Policies

## Purpose

These baseline policies exist to validate the live observation and action
pipeline before any learning agent is added.

They are intended for smoke runs, rollout debugging, and replay generation over
the CommunicationMod bridge.

## `RandomLegalPolicy`

- samples uniformly from the encoded observation's legal actions
- useful for legality smoke tests and bridge/control-loop validation
- does not try to make progress, survive, or pick sensible targets

## `SimpleHeuristicPolicy`

- uses the encoded observation plus canonical action IDs only
- outside combat, prefers `proceed`, then `choose:0`, then `leave`
- in combat, prefers a defensive card when the player is below half health and
  visible incoming damage exceeds current block
- otherwise prefers a targeted attack against the lowest-HP targetable enemy
- otherwise plays the first legal non-targeted card
- otherwise ends turn

## Non-Goals

- no learning, search, or deck-specific planning
- no attempt to solve every screen or card interaction
- no simulator-only shortcuts or direct use of raw protocol payloads
