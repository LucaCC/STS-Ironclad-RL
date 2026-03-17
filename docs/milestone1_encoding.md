# Milestone 1 Action/Observation Contract

This repo exposes a deliberately small training contract for the first
deterministic combat slice.

## Action schema

Stable action indices:

| Index | Meaning |
| --- | --- |
| 0 | `attack` |
| 1 | `defend` |
| 2 | `end_turn` |

Indices are fixed and do not depend on hand position.

## Legal action masking

The legal action mask is a length-3 boolean tuple aligned with the action indices
above.

- `attack` is legal only when energy is positive and at least one `strike` is in hand
- `defend` is legal only when energy is positive and at least one `defend` is in hand
- `end_turn` is always legal

Selecting an illegal indexed action raises `InvalidActionError`.

## Observation schema

Observations are a flat integer tuple in this exact order:

1. `player_hp`
2. `player_max_hp`
3. `player_block`
4. `energy`
5. `turn`
6. `enemy_hp`
7. `enemy_max_hp`
8. `enemy_block`
9. `enemy_intent_damage`
10. `hand_strike_count`
11. `hand_defend_count`
12. `draw_strike_count`
13. `draw_defend_count`
14. `discard_strike_count`
15. `discard_defend_count`
16. `draw_pile_count`
17. `hand_count`
18. `discard_pile_count`
19. `exhaust_pile_count`

This contract intentionally excludes unsupported mechanics and full-game state.
