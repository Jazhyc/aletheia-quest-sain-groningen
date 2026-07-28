# sonic v5 — escalation to the tested model on disagreement rows

**Submitted 2026-07-28.  Headline AUROC 0.8480 / BA 0.7950.  Regression:
−0.0581 AUROC vs v4 (0.9061).  The all-or-nothing escalation across all three
families is too aggressive — Nemotron and gemma self-reads are near-chance
on base-model data — but the Qwen big judge works (Notus flat, Iris Qwen
+0.0080).  A selective Qwen-only escalation is the next experiment.**
See `docs/official_submissions.md` § "Sonic v5 counted detail".

This document uses Simplified Technical English. The terms are the same as in
`sonic_v4.md` — "the judge" is Phoenix Wright v4 (Qwen3.5-9B + LoRA),
unchanged from v4, and it scores every row. One term is new: **the big
judge** — the tested model itself (27B or 120B), prompted to judge the row it
generated.

## 1. The change

The gate keeps both branches. The disagreement branch stops being dead.

```
# v4  (probe silenced on disagreement)   submitted, headline 0.9060
# v5  (big judge on disagreement)        this plan

# unchanged from v4 — cell 10 scores each probe, standardises it against its
# own frozen per-family mean/sd, and averages the two z-scores
probe_z = (z_L46 + z_L40) / 2

# unchanged from v4 — the judge is Phoenix Wright v4 (Qwen3.5-9B +
# LoRA). Cell 11 appends "Prediction:" to its prompt and reads the next-token
# logits of the two single tokens "0" and "1" in one forward pass.
judge_margin = logit("1") − logit("0")
judge_z      = judge_margin / JUDGE_MARGIN_SD

# unchanged from v4 — the gate runs on every row exactly as before
agree = (judge_z × probe_z > 0)
cap   = BASE_CAP + agree × (MAX_CAP − BASE_CAP)
score = sigmoid(judge_z + cap × probe_z)

# new — on the escalated rows the big judge REPLACES the judge
big_z      = big_margin[esc] / JUDGE_MARGIN_SD <---- this is a bit iffy, maybe calculate for model specifically or remove standardization or maybe it doesnt actually matter
score[esc] = sigmoid(big_z + BASE_CAP × probe_z[esc])

JUDGE_MARGIN_SD = 1.19976   # one judge step = 0.10419 z
BASE_CAP        = 2 steps = 0.20838
MAX_CAP         = 4 steps = 0.41675
# v5 adds no constant
```

`esc` is the disagreement rows — all of them, or none (§5). Two cases, and
only the second differs from v4:

| rows | score |
| --- | --- |
| agreement | `judge_z + MAX_CAP × probe_z` — v4 |
| disagreement | `big_z + BASE_CAP × probe_z` |

The judge is removed from the escalated rows entirely; the probe term
and the gate's cap decision are untouched. `big_margin` is produced the same
way as `judge_margin` — the same `Prediction:` prompt, the same two label
tokens, the same `logit("1") − logit("0")` read — only on the tested model
instead of the 9B. So it goes over the same `JUDGE_MARGIN_SD`, and nothing is
recentred or rescaled.

**If the escalation does not run, the notebook scores exactly as v4.** That is
the failure mode of every guard in §5, and it is pinned by a test.

Everything else is v4's: the dual probe, its standardization, the judge
on all other rows, the judge prompt, `JUDGE_MARGIN_SD`, both caps,
`PROBE_GAIN`, both thresholds.

## 2. Why the disagreement branch

v4 throttles the probe to `BASE_CAP` when the two detectors point opposite
ways. That branch carries no decision — it is a fallback to the judge.

It is also where the gap lives. Two units hold two-thirds of the −0.0391
deficit to Tiresias (`sonic_v4.md` §5.2): Notus Qwen (−0.0150, 38%) and Iris
Gemma (−0.0113, 29%). On Notus Qwen the judge alone scores 0.8458 and the
probe is neutral (−0.0031). No gate can fix that — a gate only redistributes
weight between two detectors that are both weak. The row needs a third reading.

The tested model is that reading: 27B or 120B against the judge's 9B, on a task
that is largely factual.

## 3. Why the trigger is disagreement

The alternative is the probe's own confidence. v4.1 gated on
`sigmoid(abs(probe_z))` and lost −0.0057 headline, Notus Nemotron −0.0380
(`sonic_v4_1.md` §4): the probe is **spuriously confident on Notus**. Anything
derived from `abs(probe_z)` inherits that. Raw probe logits are worse — the
per-family means differ by 6 logits.

Disagreement costs nothing (already computed in cell 12), and it self-selects
the right regime: on Metis and Eunomia the judge is at 0.98–0.99 so it barely
fires; on Notus both detectors are weak so it fires often.

## 4. The divisor on `big_z`

`big_z = big_margin / JUDGE_MARGIN_SD` is not a calibration. `JUDGE_MARGIN_SD`
(1.19976) is the spread of the **Phoenix 9B's** margin measured on dev. There
is no reason a 120B Nemotron's `logit("1") − logit("0")` has the same spread —
larger models generally run more confident, so the true divisor is probably
bigger and `big_z` will come out too wide. It is the only defensible constant
available without measuring, and both quantities are the same kind of read,
but that is the whole justification.

**It is not necessary for ordering.** Dividing by a positive constant is
monotone, so within the escalated rows the ranking is identical with or
without it. What it does set is two ratios: how wide the escalated group
spreads against the rest of the dataset (which is on `judge_z`'s scale), and
how much `BASE_CAP × probe_z` still counts on those rows. AUROC is computed
over all rows at once, so both are real.

How much being wrong costs, on the synthetic Notus. `k` is the factor by which
the scale is off; `k = 1` is the assumption the notebook makes:

| big judge | k=0.25 | k=0.5 | **k=1** | k=2 | k=4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.70 | −0.0544 | −0.0404 | **−0.0372** | −0.0529 | −0.0771 |
| 0.85 | −0.0257 | −0.0037 | **+0.0108** | +0.0095 | −0.0038 |
| 0.95 | +0.0024 | +0.0285 | **+0.0500** | +0.0577 | +0.0534 |

Flat-topped between k=0.5 and k=2 — being off by 2× either way costs little.
Off by 4× costs about as much as the design gains.

**The offset is the sharper risk, not the scale.** Give the big judge's margin
a systematic +1.0 against the judge's on those rows and the 0.85 row
goes from **+0.0108 to −0.0064**. The escalated rows are selected by detector
conflict, so their label mix is skewed; a constant shift moves that whole
group through the dataset ranking.

Both are one measurement: `big_margin`'s mean and sd against `judge_margin`'s
on the same dev rows. If the sd ratio is inside ~2× and the means are close,
the divisor is fine as it stands. Nothing in the notebook needs to change
until that says otherwise.

## 5. Budget

Each dataset gets **30 minutes**. If the notebook is still running when that
runs out, the sandbox kills it — and one killed dataset fails the whole
submission, not just that dataset.

The only guard is fail-soft: any error and the notebook scores exactly as v4.


## 6. Build

```bash
python experiments/ensemble_gate_eval/build_sonic_v5_notebook.py
pytest experiments/ensemble_gate_eval/test_sonic_v5_notebook.py
python submit.py --dry              # not yet run
```

The source is `legacy_submissions/sonic_v4.ipynb`, which the build script
edits in three places:

- **cell 9** records `extract_seconds`, so §5's cost projection uses a
  measurement from the same run on the same model rather than a guess;
- **a new cell 12** is inserted — the escalation, producing `big_margin` and
  `esc_mask`;
- **cell 13** (v4's cell 12) swaps `judge_z` for `big_z` on those rows.

The other 10 cells are asserted byte-identical to v4. The constants are
`results/ensemble_gate_eval/sonic_v5_constants.json`.

The offline go/no-go measurement has **not** been built or run.
