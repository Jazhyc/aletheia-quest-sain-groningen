# sonic v4.2 -- dual-probe (L40 + L46), judge-uncertainty gate

**Built 2026-07-27, submitted 2026-07-27 18:00, scored AUROC `0.9047` / BA `0.8183` — regression (−0.0013 vs v4). Notus Nemotron −0.0147. Moved to `legacy_submissions/`.**

This document uses Simplified Technical English. The terms are the same as in
`sonic_v4.md`.

## 1. The change

v4's gate opens the cap on agreement and throttles to BASE_CAP on
disagreement. v4.2 adds one exception: on disagreement rows where the judge
is *uncertain*, the cap opens.

```
# v4   (sign test, binary)
# v4.2 (sign test + judge-uncertainty exception)   this notebook

agreement  = (judge_z * probe_z > 0)
judge_uncertain = sigmoid(-abs(judge_z) + 1.5)

cap = BASE_CAP + (agreement + (1 - agreement) * judge_uncertain) * (MAX_CAP - BASE_CAP)
score = sigmoid(judge_z + cap * probe_z)

BASE_CAP = 0.20838  (2 steps)
MAX_CAP  = 0.41675  (4 steps)
```

The dual probe (L40 + L46), the judge, and every other constant are v4's,
unchanged. Two executable lines differ from v4.

## 2. Why judge uncertainty

v4.1 failed because it decoupled the cap from the judge entirely -- the
probe got through on Notus rows where it was confident but wrong, and the
judge's ranking was damaged (-0.0380 Notus Nemotron).

v4.2 keeps the judge coupling but adds a narrow exception: the cap opens on
disagreement rows only when the judge is *uncertain* (|judge_z| < ~1.5). On
dev data, 98% of disagreement rows have |judge_z| < 1.0 -- the sign test
blocks the probe on exactly the rows where the judge barely has an opinion.

| row | judge | v4 cap | v4.2 cap |
| --- | --- | ---: | ---: |
| Agree | any | 4 steps | 4 steps (unchanged) |
| Disagree, judge sure | \|jz\| > 2 | 2 steps | 2 steps (unchanged) |
| Disagree, judge uncertain | \|jz\| < 1 | 2 steps | 3.2-3.6 steps |

On Iris, the uncertain disagreement rows are where the probe is right and
would help most. On Notus, the uncertain disagreement rows are where the
judge itself is near-chance -- there is no strong ranking to damage, and
the probe's random nudges add bounded noise.

Judge-confident rows are untouched at 2 steps. This preserves v4's Notus
safety: the rows that anchor the ranking stay exactly as they were.

## 3. Cap values on disagreement rows

| \|judge_z\| | judge_uncertain | cap | cap (steps) |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.82 | 0.379 | 3.6 |
| 0.5 | 0.73 | 0.360 | 3.5 |
| 1.0 | 0.62 | 0.337 | 3.2 |
| 1.5 | 0.50 | 0.312 | 3.0 |
| 2.0 | 0.38 | 0.287 | 2.8 |
| 3.0 | 0.18 | 0.246 | 2.4 |
| 4.0 | 0.08 | 0.224 | 2.2 |

The transition is smooth. The cap exceeds 3 steps only when |judge_z| < 1.5.

## 4. Decision tree

`judge_z * probe_z` is element-wise (per-row) multiplication, **not** a dot
product. It checks whether the judge and probe point the same direction *on
that specific row*. Positive → same sign (agree), negative → opposite signs
(disagree).

```
For each row:

  judge_z * probe_z > 0 ?
     |
     +-- YES (agree: same sign)
     |       cap = MAX_CAP  (4 steps)
     |       The probe amplifies the judge's direction.  Same as v4.
     |
     +-- NO  (disagree: opposite signs)
            |
            |judge_z| < ~1.5 ?
               |
               +-- YES (judge uncertain: barely has an opinion)
               |       cap = ~3.2-3.6 steps  (near MAX_CAP)
               |       **CHANGED from v4** (v4 gave 2 steps here).
               |       The probe can nudge the score despite disagreeing
               |       with the judge.  On Iris the probe is right on
               |       these rows; on Notus the judge is near-chance,
               |       so probe noise is bounded.
               |
               +-- NO  (judge certain: has a strong opinion)
                       cap = BASE_CAP  (2 steps)
                       The probe is throttled.  Same as v4.
                       These rows anchor the ranking -- not touched.
```

Same logic as a table:

| judge_z × probe_z > 0? | \|judge_z\| | v4 cap | v4.2 cap | |
| --- | --- | ---: | ---: | --- |
| Yes | any | 4 steps | 4 steps | |
| No | < 1.5 | 2 steps | **3.2-3.6** | |
| No | > 1.5 | 2 steps | 2 steps | |

The only row that changed from v4 is the middle one. On dev data, 98% of
disagreement rows have |judge_z| < 1.0. v4 was throttling the probe to 2
steps on nearly every disagreement row; v4.2 opens to ~3.5 steps on nearly
all of them.

## 5. Build

```bash
python experiments/ensemble_gate_eval/build_sonic_v4_2_notebook.py
```

The source was `submission/sonic_v4.ipynb`. The build script patches cell 12
only. The remaining 12 cells are byte-identical to v4. v4 has since been
restored to `submission/` as the best submitted AUROC.

## 6. Risks

The same three as in `sonic_v4.md` section 7 apply: the Notus proxy is too
optimistic, the probe is in-sample on dev folds, part of the Iris shortfall
is unexplained.

One new risk: on Notus disagreement rows where the judge is uncertain, the
cap opens to ~3 steps and a random probe signal gets through. v4 kept these
at 2 steps. The judge is near-chance on these rows, so the noise is bounded,
but it is non-zero. The v4.1 collapse (-0.0380 Notus Nemotron) was caused
by opening the cap on judge-*confident* rows -- v4.2 does not touch those.

The mechanism was verified on local dev data: 98% of disagreement rows have
|judge_z| < 1.0, and the judge is near-chance on those rows (mean |judge_z|
= 0.27 on varied data). But dev is not Notus -- the actual proportion of
uncertain disagreement rows on Notus is unknown.
