# Milestone 1 Trainable Environment Slice

This document defines the first trainable combat environment slice for the RL stack.
It is intentionally narrower than full *Slay the Spire* combat and should align with the current deterministic combat foundation in `src/sts_ironclad_rl/env/`.

## Scope

Milestone 1 training should use a single deterministic combat with:

- one Ironclad starter-deck subset
- one enemy profile
- one combat at a time
- no map, rewards screen, relics, potions, or progression systems

The purpose of this slice is to validate environment wrappers, action masking, observation encoding, reproducible rollouts, and a first end-to-end training loop on a tiny but non-trivial decision problem.

## Supported Cards

The first slice supports only these player cards:

- `strike`
  - cost: 1 energy
  - effect: deal 6 damage to the enemy
- `defend`
  - cost: 1 energy
  - effect: gain 5 block

The initial training deck should be the currently implemented five-card subset:

- `("strike", "strike", "strike", "defend", "defend")`

No card upgrades, exhaust behavior, draw effects, status cards, or additional starter cards are included in this slice.

## Supported Enemy Set

The first slice supports exactly one enemy profile.

- single unnamed training enemy
- fixed max HP from `EncounterConfig`
- no block, buffs, debuffs, or multi-enemy coordination
- deterministic attack-only pattern driven by `_enemy_damage(seed, turn)`

Current implemented enemy pattern:

- enemy deals `6 + ((seed + turn) % 2)` damage on its turn
- this yields a deterministic 6/7 alternating pattern for a fixed seed

This should be treated as a simple scripted opponent for training infrastructure, not as a faithful reproduction of a real *Slay the Spire* enemy.

## Supported Mechanics

The slice includes only the following mechanics:

- deterministic seed-based deck shuffle at reset
- draw pile, hand, discard pile, and exhaust pile state containers
- initial draw on reset
- per-turn energy
- legal action masking
- card play for supported cards
- enemy HP reduction
- player block gain and damage prevention
- end-turn hand discard
- reshuffle when the draw pile is empty and the discard pile must become the next draw pile
- deterministic enemy turn
- terminal combat resolution on player death or enemy death

## Unsupported Mechanics

The following are explicitly out of scope for Milestone 1:

- full Ironclad starter deck beyond `strike` and `defend`
- additional cards such as `bash`
- vulnerable, weak, frail, strength, dexterity, or any other buffs/debuffs
- card upgrade paths
- exhaust effects as gameplay, even though an exhaust pile container exists
- variable-cost cards, X-cost cards, or zero-cost cards
- enemy intents beyond direct damage
- multiple enemies
- stochastic combat events beyond seed-controlled setup and enemy pattern selection
- relics, potions, gold, rewards, shops, events, map pathing, or act progression
- out-of-combat healing
- save/load mid-episode state APIs
- reward shaping tied to hand quality, block amount, or intermediate heuristics

## Episode Start Conditions

Each episode starts from `CombatEnvironment.reset(seed)` with a fixed `EncounterConfig`.

Required start conditions:

- player HP is full: `player.hp == player.max_hp`
- enemy HP is full: `enemy.hp == enemy.max_hp`
- player block is 0
- enemy block is 0
- turn is 1
- energy is `starting_energy`
- deck order is a deterministic shuffle from the provided seed
- opening hand is drawn immediately after reset

For the first training pass, training code should treat the encounter configuration as fixed across episodes except for the reset seed.

## Terminal Conditions

An episode terminates when either of these becomes true:

- enemy HP reaches 0
- player HP reaches 0

No truncation rules are required for the first pass, though a wrapper may optionally add a defensive max-step limit to catch integration bugs.

## Reward Definition

The first training pass should use sparse outcome reward at episode end:

- win: `+1.0`
- loss: `-1.0`
- all non-terminal transitions: `0.0`

This is the recommended wrapper-level reward even though the current `CombatEnvironment.step()` returns immediate `+6.0` reward on `attack`. The environment core can stay as-is for now, but the trainable wrapper should override reward semantics to the sparse terminal objective above so the spec stays narrow and easy to reason about.

## Action Representation Assumptions

The training wrapper should expose a fixed discrete action space with stable indices.

Recommended mapping:

- `0 -> attack`
- `1 -> defend`
- `2 -> end_turn`

Assumptions:

- actions are card-category actions, not per-card-instance actions
- if multiple copies of a supported card are in hand, playing that action consumes one copy
- illegal actions must be masked from the policy
- if an illegal action still reaches the environment, the wrapper should either reject it deterministically or convert it into a hard error during development

## Observation Representation Requirements

The first trainable slice needs a deterministic, fixed-size observation encoding. At minimum, the wrapper should expose:

- player HP
- player max HP
- player block
- enemy HP
- enemy max HP
- current energy
- current turn
- counts of `strike` in hand, draw pile, and discard pile
- counts of `defend` in hand, draw pile, and discard pile
- total cards remaining in draw pile
- total cards in hand
- total cards in discard pile
- current enemy incoming damage for this turn
- legal action mask

Observation requirements:

- encoding must be fixed-shape across all states
- field order must be documented and stable
- observation generation must be a pure function of state
- no hidden RNG or wrapper-local latent state may affect observations

## Determinism And Reproducibility

Milestone 1 training runs must be reproducible from explicit seeds.

Requirements:

- reset with the same seed and config must produce the same initial state
- applying the same action sequence from the same initial state must produce the same trajectory
- action masks and observations must be deterministic functions of state
- reward calculation in the trainable wrapper must be deterministic
- training code should separate environment seeds from model initialization seeds
- tests should cover deterministic reset and at least one deterministic rollout

## Implementation Gaps

The current combat foundation is close to this slice, but wrapper and training threads will still need a few hooks:

- fixed discrete action index mapping exported alongside `Action`
- observation encoder that converts `CombatState` into a documented fixed-size tensor or array
- wrapper-level sparse terminal reward adapter
- helper for current enemy intent or incoming damage so observations do not need to reach into private combat helpers
- reset/step API shape that returns observation, reward, done, and optional metadata for training code
- optional episode summary info such as win/loss, turns survived, and final HP values for logging
