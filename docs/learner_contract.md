## Learner Contract

This document freezes the training-facing interface consumed by the current
masked-DQN baseline and intended future DQN-family upgrades.

Source of truth:
[`sts_ironclad_rl.training.learner`](/Users/lucacc/Desktop/STS/STS-Ironclad-RL/src/sts_ironclad_rl/training/learner.py)

## Current Use

The contract is used by:

- `LearnerObservationEncoder`
- `LearnerActionIndex`
- `LearnerTransitionExtractor`
- `ReplayBuffer`
- `MaskedDQN`
- `DQNTrainer`

It is built on top of shared live replay output. It does not introduce a second
rollout loop.

## Observation Vector

Default layout:

- `vector_size = 93`
- schema version: `learner_observation.v1`
- max hand slots: `10`
- max enemies: `5`

Feature ordering is fixed and deterministic:

1. player hp, max hp, block, energy
2. draw, discard, and exhaust sizes
3. turn index
4. hand slot masks plus per-slot card features
5. enemy slot masks plus per-slot enemy features

Padding and truncation rules:

- only the first 10 hand slots are represented
- only the first 5 enemies are represented
- padded slots use zeroed features and mask `0`
- dead but still-present enemies remain represented with `alive = 0`

## Action Index Space

Default layout:

- `action_space_size = 61`
- `0 -> END_TURN`
- `1..10 -> PLAY_CARD_HAND_i_NO_TARGET`
- `11..60 -> PLAY_CARD_HAND_i_TARGET_ENEMY_j`

Out-of-bounds live actions are intentionally not representable in the learner
space. Replay extraction raises if a stored replay action cannot be mapped into
the frozen bounds.

## Legal Mask

- mask length is always `61`
- mask ordering exactly matches the action index space
- stored transition `mask` is the legal-action mask for `next_state`
- malformed or degenerate masks are sanitized at policy/runtime boundaries, not
  by changing the frozen schema

That next-state mask choice is deliberate because masked DQN needs it for the
bootstrap target.

## Reward Function

Default shaped reward:

```text
reward =
  + 0.1 * max(0, enemy_hp(s) - enemy_hp(s'))
  - 0.2 * max(0, player_hp(s) - player_hp(s'))
  - 0.01
  + 1.0 if terminal victory
  - 1.0 if terminal defeat
```

`combat_end` does not receive an extra terminal bonus.

Reward weights live in `RewardConfig`, so reward shaping can change without
changing the transition schema.

## Transition Schema

`LearnerTransition` stores:

- `state`
- `action_index`
- `reward`
- `next_state`
- `done`
- `mask`

Tuple order:

```text
(state, action_index, reward, next_state, done, mask)
```

Replay extraction rules:

- one transition is emitted for each replay entry with an action that is
  followed by another replay entry
- terminal-only replay entries do not emit transitions on their own
- extraction consumes the same replay entries written by the shared live rollout path

## Stability Promise

This contract is frozen for the current trainable-agent milestone. Future
algorithm work should extend around it where possible instead of redefining the
observation vector, action space, or transition tuple.
