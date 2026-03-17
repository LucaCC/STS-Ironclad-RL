# Current Milestone

## Active Focus

Milestone 2: first trainable live masked-DQN baseline

## Goal

Keep the repo ready for repeated live experiments on one coherent baseline path:

`live game -> rollout -> learner contract -> replay -> DQN training -> checkpoint -> evaluation -> benchmark`

## Done In This Milestone

- shared live rollout, evaluation, and collection path
- frozen learner contract
- replay-backed masked-DQN trainer with checkpointing
- baseline-vs-DQN benchmark workflow
- canonical configs, scripts, artifacts, and docs for the first experiment loop

## Success Criteria

- contributors can run the baseline benchmark, train the baseline DQN, evaluate
  the checkpoint, and compare it against the baselines without guessing file paths
- non-live tests stay deterministic
- docs describe the implemented system honestly, including current limits
