## Learner Contract

This document freezes the training-facing interface that future masked-DQN code
should consume. It is built on top of the existing live rollout path and does
not introduce a second execution loop.

All defaults below come from
[`sts_ironclad_rl.training.learner`](/Users/lucacc/Desktop/STS/STS-Ironclad-RL/src/sts_ironclad_rl/training/learner.py).

## Observation Vector

Default observation layout:

- `MAX_HAND_SLOTS = 10`
- `MAX_ENEMIES = 5`
- `vector_size = 93`
- schema version: `learner_observation.v1`

Feature ordering is fixed and deterministic:

1. `player_hp`
2. `player_max_hp`
3. `player_block`
4. `player_energy`
5. `draw_pile_size`
6. `discard_pile_size`
7. `exhaust_pile_size`
8. `turn_index`
9. `hand_mask_0 .. hand_mask_9`
10. `hand_0_card_token, hand_0_cost, hand_0_is_playable, hand_0_has_target`
11. `hand_1_*`
12. `...`
13. `hand_9_*`
14. `enemy_mask_0 .. enemy_mask_4`
15. `enemy_0_current_hp, enemy_0_max_hp, enemy_0_block, enemy_0_intent_token, enemy_0_alive, enemy_0_targetable`
16. `enemy_1_*`
17. `...`
18. `enemy_4_*`

Padding and truncation rules:

- Hand cards are kept in bridge order and truncated to the first 10 cards.
- Enemies are kept in bridge order and truncated to the first 5 enemies.
- `hand_mask_i = 1` when slot `i` is populated and `0` when the slot is padded.
- `enemy_mask_i = 1` when slot `i` is populated and `0` when the slot is padded.
- Dead enemies are still populated slots when present in the live snapshot. They
  are not padding. Their `enemy_i_alive` feature is `0`.
- Padded card and intent tokens use `0.0`.

Categorical token fields:

- `card_token` is a deterministic SHA-1 derived numeric token from the live
  `card_id` when available, else the card name.
- `intent_token` is a deterministic SHA-1 derived numeric token from the live
  enemy intent string.

## Action Index Space

Default action layout:

- `MAX_HAND_SLOTS = 10`
- `MAX_ENEMIES = 5`
- `action_space_size = 61`

Index mapping:

- `0 -> END_TURN`
- `1 .. 10 -> PLAY_CARD_HAND_i_NO_TARGET` for hand slots `0 .. 9`
- `11 .. 60 -> PLAY_CARD_HAND_i_TARGET_ENEMY_j` in row-major order:
  `11 + i * MAX_ENEMIES + j`

Examples:

- `1 -> PLAY_CARD_HAND_0_NO_TARGET`
- `10 -> PLAY_CARD_HAND_9_NO_TARGET`
- `11 -> PLAY_CARD_HAND_0_TARGET_ENEMY_0`
- `15 -> PLAY_CARD_HAND_0_TARGET_ENEMY_4`
- `16 -> PLAY_CARD_HAND_1_TARGET_ENEMY_0`

Out-of-bounds behavior:

- Cards beyond the first 10 hand slots are not representable in the learner
  action space.
- Enemies beyond the first 5 slots are not representable in the learner action
  space.
- Replay extraction raises if a stored action references an out-of-bounds hand
  or enemy index.

## Legal Action Mask

The legal mask length is always `61` and aligns exactly with the action index
ordering above.

Mask semantics:

- `1` means the learner action index is legal for that snapshot.
- `0` means it is illegal.
- `END_TURN` is marked legal for combat snapshots.
- Play-card legality is inherited from the existing live action contract:
  `is_playable`, target requirements, targetability, and any live-side energy
  constraints already reflected by `is_playable`.
- Unrepresentable actions outside the learner bounds are dropped from the mask.

Within extracted learner transitions, `mask` is the legal mask for `s'`
(`next_state`). That is the mask needed for masked-DQN bootstrap targets.

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

`combat_end` terminal states do not receive an extra terminal bonus.

All reward weights live in `RewardConfig` and can be changed without changing
the transition schema.

## Transition Schema

`LearnerTransition` stores:

- `state`: learner observation vector for `s`
- `action_index`: discrete learner action index for `a`
- `reward`: shaped scalar reward
- `next_state`: learner observation vector for `s'`
- `done`: terminal flag
- `mask`: legal action mask for `s'`

`LearnerTransition.as_tuple()` returns:

```text
(s, a, r, s', done, mask)
```

Replay extraction rules:

- One transition is emitted for each replay entry that contains an action and is
  followed by another replay entry.
- Terminal-only replay entries do not emit transitions by themselves.
- The extractor consumes the same replay entries produced by the shared live
  rollout path.
