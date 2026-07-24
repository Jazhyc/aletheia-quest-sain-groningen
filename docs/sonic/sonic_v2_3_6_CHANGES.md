# sonic_v2.3.6

This document tells you what sonic_v2.3.6 changes over sonic_v2.3.5. Read
`docs/sonic/sonic_v2_3_CHANGES.md` first — it describes the shared gate that both
versions run. Read `docs/sonic/sonic_v2_3_6_EXPLAINER.md` for the full step-by-step
walkthrough and the design critique this version acts on.

**In one line:** sonic_v2.3.6 keeps the whole sonic_v2.3.5 detector and adds two
gate-robustness guards. It does not touch the probe, the judge model, the judge
prompt, or the rank-blend math. The submission notebook is
`submission/sonic_v2.3.6.ipynb`.

---

## Why we made v2.3.6

sonic_v2.3.5 is a robust combiner of a white-box probe and a black-box judge.
A close read of the gate found two places where the combiner can misfire on
data it has never seen. Neither shows up on the dev datasets, because both are
triggered by conditions the dev datasets do not contain. Both changes are
conservative: each reduces to the exact v2.3.5 behaviour outside a narrow edge
case, so neither can regress a healthy dataset.

---

## The two detectors it still uses

Unchanged from v2.3.5.

- The **probe** reads the model's activations at decoder layer 46 and gives a
  score in `[0, 1]`. It is white-box and needs weights for the dataset's base
  model. It generalizes well in distribution and collapses to chance on novel
  organisms (it sat at chance on every Notus unit).
- The **judge** reads only the conversation text. It is `Qwen3.5-9B` with a LoRA
  adapter. It gives a hard `0/1` verdict and a soft confidence in `[0, 1]`.
  Because it is black-box, it runs even on a base model that has no probe
  weights.

---

## Change A — the judge covers unseen base models

**The fault.** In v2.3.5 the judge runs only when the dataset name starts with
`validation-`, or when `SONIC_FORCE_JUDGE=1`:

```python
RUN_JUDGE = IS_VALIDATION_SPLIT or FORCE_JUDGE
```

Consider a hidden dataset that is **both** on a base model with no probe weights
**and** not named `validation-*`. The probe cannot run (no weights). The judge
does not run (the name gate is false). The gate then reaches the
`not probe_ok and not judge_ok` branch and writes `0.5` on every row — the worst
possible output, on exactly the kind of novel data the judge exists to carry.
This is a dependency on a dataset-naming convention we do not control at test
time, standing in a load-bearing spot.

**The fix.** Also run the judge whenever the base model has no probe weights:

```python
RUN_JUDGE = IS_VALIDATION_SPLIT or FORCE_JUDGE or (base_model is None)
```

**Why it is safe.** This only *adds* a judge run in the one case that currently
returns an all-`0.5` write. It cannot change any dataset that already has a
probe, and every dev dataset has a probe, so the change is invisible on dev and
cannot regress it. It is insurance against an assumption we cannot verify until
the hidden test.

**Cost.** It adds one judge pass on a no-probe dataset. The budget-aware retry
logic (`NOTEBOOK_BUDGET_SECONDS`, unchanged) already guards the wall-clock
limit, so the extra pass cannot push the run over the timeout.

---

## Change C — do not invert near-chance agreement

**The fault.** The gate measures `agreement = AUROC(judge verdicts, probe
scores)` and, if it falls below `0.5`, assumes the probe ranks consistently
backwards and flips it:

```python
if agreement < 0.5:
    agreement = 1.0 - agreement
    probe_for_gate = 1.0 - probe_scores
w = clip(2 * agreement - 1, 0, W_MAX)
```

Agreement is an AUROC computed on one dataset's rows, often with few judge
positives, so it is a noisy estimate. A value of `0.47` is far more likely to be
noise around chance than a genuinely inverted probe. The old code inverts it to
`0.53` and feeds `w = 0.06` of **wrong-direction** probe signal into the blend.

**The fix.** Only invert when agreement is below chance by a margin; inside the
near-chance band, drop the probe from the blend instead of flipping it.

```python
INVERT_MARGIN = float(os.environ.get("SONIC_INVERT_MARGIN", "0.05"))
if agreement < 0.5 - INVERT_MARGIN:
    agreement = 1.0 - agreement            # genuine backwards signal: flip it
    probe_for_gate = 1.0 - probe_scores
elif agreement < 0.5:
    agreement = 0.5                        # near-chance noise: w -> 0, drop the probe
w = clip(2 * agreement - 1, 0, W_MAX)
```

**Why it is nearly free.** For agreement `>= 0.5` the code is identical to
v2.3.5. For agreement `< 0.45` (a clearly inverted probe) it is identical to
v2.3.5. Only the band `[0.45, 0.5)` changes: there the probe weight goes to `0`
(judge alone) instead of a tiny positive value pointing the wrong way. The
verified `w` mapping:

| agreement | 0.30 | 0.44 | 0.46 | 0.47 | 0.49 | 0.50 | 0.55 | 0.80 |
|-----------|------|------|------|------|------|------|------|------|
| w (v2.3.5)| 0.40 | 0.12 | 0.08 | 0.06 | 0.02 | 0.00 | 0.10 | 0.60 |
| w (v2.3.6)| 0.40 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.60 |

**Honesty note.** This is *nearly* free, not strictly free. A probe that is
genuinely, slightly inverted (true agreement in `[0.45, 0.5)`) loses its small
contribution. We judge that trade clearly positive — near-chance agreement is
almost always noise — but it is a trade, not a theorem.

---

## What v2.3.6 deliberately does NOT change

- **The probe.** Same weights, same layer, same tokenization.
- **The judge.** Same model, same LoRA adapter, same prompt, same hard-verdict
  and soft-token readout.
- **The rank blend.** `_rank01`, the blend `w * probe_rank + (1 - w) *
  judge_rank`, and the prevalence-matched threshold are untouched.
- **The `W_MAX = 0.9` cap.** We argued elsewhere it is marginal insurance, but
  changing it is a trade-off we cannot measure without the held-out judge-output
  experiment. We do not tune it blind.
- **Change B (small-sample trust shrinkage).** We designed a third change that
  shrinks `w` toward the judge when the judge has few positives, to damp the
  noisy-agreement problem more aggressively than change C. It is **not** in
  v2.3.6. Unlike A and C it is a real bias/variance trade that can hurt a
  small-sample dataset on which the probe is actually right, so it must be
  validated on the leakage-free holdout before shipping. See the EXPLAINER.

---

## Environment variables added

- `SONIC_INVERT_MARGIN` (default `0.05`) — how far below `0.5` agreement must
  fall before the probe is treated as inverted rather than as near-chance noise.

All v2.3.5 variables (`NOTEBOOK_BUDGET_SECONDS`, `JUDGE_RETRY_MARGIN_SECONDS`,
`SONIC_FORCE_JUDGE`, `SUBMISSION_THRESHOLD`, …) are unchanged.

---

## What to expect

**Baseline (confirmed).** sonic_v2.3.5 was submitted on 2026-07-24 19:22 and
scored BA `0.853750` / AUROC `0.876333` — our best official submission on both
metrics. That is the number v2.3.6 inherits and must not regress.

On the dev datasets: **no change.** Every dev dataset has probe weights (change
A is inert) and none produce a near-chance-but-inverted agreement (change C is
inert). The dev balanced accuracy and AUROC should match sonic_v2.3.5 exactly.

The value of v2.3.6 is entirely on hidden data:

- Change A turns a guaranteed `0.5` into a real judge score on any hidden
  dataset whose base model has no probe and whose name does not start with
  `validation-`.
- Change C removes a small wrong-direction contribution on any hidden dataset
  where the probe/judge agreement lands just under chance.

Both are downside protection, not ceiling raising. The probe is still the
detector that fails out of distribution, and v2.3.6 does not fix that — it only
stops two ways the combiner can mishandle that failure.
