# sonic v3.5

**Built 2026-07-27, submitted 2026-07-27. Scored AUROC `0.9046` / BA `0.8275`. Moved to `legacy_submissions/`.**

One line changes against v3.4. The cap opening becomes a direction test.

```
agreement = (judge_z × probe_z > 0)              # v3.5
agreement = clip(judge_z × probe_z / 3, 0, 1)    # v3.2 – v3.4

cap   = BASE_CAP + agreement × (MAX_CAP − BASE_CAP)
score = sigmoid(judge_z + cap × tanh(PROBE_GAIN × probe_z))
```

AGREEMENT_SCALE is deleted. BASE_CAP stays at 2 steps, MAX_CAP at 4. The probe,
the judge, the prompt and the threshold are v3.4's. The pipeline cells are
byte-identical to v3.4 and the scoring cell differs in three executable lines.

## What was wrong with the product

The product opening correlates **+0.94 with |judge_z|**. Sort the 8000 dev rows
by judge confidence: the least certain fifth opens the cap 3%, the most certain
fifth opens it 78%.

AUROC is pairwise ordering. A row the judge is sure about sits far from the
ordering boundary, so moving it cannot change many pairs. A row the judge is
unsure about sits in the middle, where the pairs are. So the product gave the
probe its loudest vote where it could not matter, and muted it where it could.

## Why the product was not simply a mistake

That same coupling is what kept Notus safe, and this is the part the first read
of the design got wrong. When the probe is noise, spending its influence on
judge-confident rows is protective: a random probe cannot reorder rows that are
far from the boundary. Gates that ignore judge confidence collapse in that
regime. Blunted to the probe quality Notus actually has, a flat cap costs
`-0.0137` mean and a judge-uncertainty gate `-0.0111`, against the product's
`-0.0039`.

The safety comes from the **direction** check, not from the magnitude. The sign
test keeps the direction check and drops the magnitude.

## The fit was calibrated to the wrong probe

`fit_bounded_refine_v3_2.py` selected v3.2's constants against a probe blunted to
AUROC 0.76, described as "the Notus regime". `sonic_v3_3_mini` has since measured
the probe's real Notus AUROC on the private split: **0.5586**, and 0.4808 on the
gemma unit, which is below chance.

The correction changes conclusions. At 0.76 a flat cap looks acceptable and the
sign test looks strictly better than the product on every axis. At 0.56 the flat
cap is a disaster and the product is the safer of the two. `fit_sign_gate_v3_5.py`
scores at 0.56 and at 0.48 inverted.

The gain side was measured wrong too. The average dev fold has the probe beating
the judge by `+0.014`. Real Iris has it beating the judge by `+0.068`. Averaging
over all 20 folds measures a condition Iris is not in, so gains here are read off
the 10 folds where the probe most outclasses the judge.

## Measured

Caps at 2 and 4 steps, 20 folds, 5 seeds.

| | product | sign |
| --- | ---: | ---: |
| Iris-like gain (10 folds, probe edge +0.0276) | +0.0150 | **+0.0183** |
| — as a share of the probe's edge | 54.3% | **66.1%** |
| Notus 0.56, mean | **−0.0039** | −0.0049 |
| Notus 0.48 inverted, mean | **−0.0089** | −0.0094 |

Paired per fold, the sign test wins 17 of 20 Iris-like folds and loses 1. It
loses 79 of 100 fold-draws at 0.56, each loss small: mean `−0.0010`, worst
`−0.0061`.

Projected headline change: **+0.0011**. That is half the Iris-like gain plus half
the Notus cost, since the headline is the mean of the two counted datasets.

## Rejected

Net is half the Iris-like gain plus half the Notus cost. The last column is what
the change is worth against the product gate we ship today.

| shape | Iris-like gain | captures | Notus 0.56 | net | vs product |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3.2 agreement (product) | +0.0150 | 54.3% | −0.0039 | +0.0056 | — |
| **sign only** | **+0.0183** | **66.1%** | **−0.0049** | **+0.0067** | **+0.0011** |
| sign × probe-confidence, k=1 | +0.0184 | 66.4% | −0.0048 | +0.0068 | +0.0012 |
| sign × uncertainty, w=2.0 | +0.0179 | 64.6% | −0.0047 | +0.0066 | +0.0010 |
| judge-uncertainty, w=1.5 | +0.0210 | 75.9% | −0.0111 | +0.0049 | −0.0006 |
| flat, always open | +0.0222 | 80.4% | −0.0137 | +0.0043 | −0.0013 |

The two shapes that capture the most of the probe's Iris edge are the two that
lose the most on Notus, and both come out behind the product gate once Notus is
counted. The three sign-based shapes are indistinguishable — they sit inside
`0.0002` of each other, which is noise at this fold count. Plain sign wins on
simplicity: the other two buy nothing and each costs a constant.

## What this is not

The gate is on its efficient frontier and this is a `+0.001`-sized move. It does
not touch the two real gaps: Notus, where the probe is useless and the judge
carries everything, and the 72% of the probe's Iris edge that the cap still
throws away. Those need a per-row estimate of probe *reliability* — an OOD score
in the probe's own feature space — or a better judge. Not a gate constant.

## Build

```bash
python experiments/ensemble_gate_eval/fit_sign_gate_v3_5.py
python experiments/ensemble_gate_eval/build_sonic_v3_5_notebook.py
pytest experiments/ensemble_gate_eval/test_sonic_v3_5_notebook.py
```

Source: `legacy_submissions/sonic_v3_4.ipynb`. Constants:
`results/ensemble_gate_eval/sonic_v3_5_constants.json`. Swapping which notebook
is queued: `legacy_submissions/README.md`.

## Pre-flight, 2026-07-27

Run locally, against the shipped cells rather than a reimplementation.

- All three probe families load from `submission/whitebox_probe/` through the
  notebook's own cells 4 and 5, layer 46, `trained_by=shared_trunk_multifamily`,
  forward pass finite. Those cells swallow load failures and fall back to
  judge-only, so the check asserts `base_model` survived rather than just
  watching for an exception.
- The scoring cell, executed on 500 synthetic rows, emits exactly two cap
  magnitudes — `0.208376` and `0.416752`, i.e. BASE and MAX — confirming the gate
  is binary on the shipped code path. Mean agreement 0.504, which is what a
  direction test should give on independent inputs.
- Scores land in (0,1) with no ties, threshold `0.20`.
- Fallbacks work: judge-only (probe missing), probe-only at threshold `0.50`
  (judge missing), and the empty dataset.
- `submit.py` packages 258 files / 11.8 MB, exactly one notebook, nothing from
  `legacy_submissions/`.

Cosmetic, pre-existing and unchanged from v3.2: on an empty dataset the log line
prints `agreement=nan`, because `np.mean` of an empty array is nan. No score is
affected.

Not covered locally: NDIF extraction and the judge pass. Only `submit.py --dry`
or the real run exercises those.
