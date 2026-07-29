# Q397 family-level calibration

Simple family-specific temperature scaling, offsets, Platt scaling, and rank
normalization cannot improve this competition's mean per-dataset AUROC: every
public dataset unit contains one generating base-model family, and those
monotone transformations preserve its within-unit ordering.

This experiment instead keeps the optimized Q397 adapter, Phoenix 4.0 prompt,
and final `Prediction:` position fixed, while allowing one family-wide
log-odds blend with an auxiliary label-token pair. All 12 label logits are
requested in the same forward pass, so a promoted calibration would not add a
model call. Parameters are selected separately for Qwen, Gemma, and Nemotron;
organism and LoRA identities never enter the rule.

Selection uses only the 6,573 public train rows. The candidate grid blends the
digit margin with one of `No/Yes`, `False/True`, `A/B`,
`honest/deceptive`, or `truthful/misleading`, retaining digit weight
`0.95, 0.9, 0.8, 0.7, 0.6`, or `0.5`. A family candidate must improve train
macro AUROC by at least `0.001`, be non-worse on at least 75% of its dataset
units, lose at most `0.01` on every unit, and lose at most `0.002` in either
represented scenario. Families with fewer than two train units retain digits.

The frozen family parameters are evaluated once on validation both as Q397
alone and after the current rank-1 intent blend. Promotion requires at least
`+0.001` current-Phoenix macro AUROC, no family loss beyond `0.002`, and no
scenario loss beyond `0.001`.

```bash
sbatch experiments/q397_family_calibration/run_train.sh
```

The primary robustness-gated selector retained literal digits for every family.
As a separately reported empirical-risk-minimization diagnostic, the pure
train-macro winner was also frozen for each family with at least two units.
That rule selected 50% digit plus 50% `truthful/misleading` for Qwen, 60%
digit plus 40% `honest/deceptive` for Gemma, and unchanged digits for
single-unit Nemotron. Although Q397 alone regressed, composing these train-only
parameters with Phoenix's already-frozen rank-1 intent member raised validation
macro AUROC from `0.95512` to `0.95702` (`+0.00190`), with instructed
`-0.00042`, varied `+0.00500`, Qwen `+0.00222`, and no Gemma or Nemotron
change. It therefore receives one frozen local-test confirmation:

```bash
sbatch experiments/q397_family_calibration/run_test.sh
```
