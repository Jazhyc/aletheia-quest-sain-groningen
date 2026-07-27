# sonic v4.1 — dual-probe (L40 + L46), confidence gate

**Built 2026-07-27, submitted 2026-07-27. Scored AUROC `0.9003` / BA `0.8154`.
Regression vs v4 (−0.0057 AUROC). Moved to `legacy_submissions/`.**

This document uses Simplified Technical English. The terms are the same as in
`sonic_v4.md`: "the probe", "the judge", "the gate", "the cap", "the step",
"the organism".

## 1. The change

One line in the gate changes. The dual probe from v4 stays.

```
# v4   (sign test)              submitted, headline 0.9060
# v4.1 (confidence gate)        this notebook

cap   = sigmoid(abs(probe_z)) × MAX_CAP
score = sigmoid(judge_z + cap × probe_z)

MAX_CAP = 0.41675  (4 steps)
```

The dual probe (L40 + L46 fused as `(z_46 + z_40) / 2.0`), the judge, the
prompt, and every constant except `BASE_CAP` (now unused) are v4's, unchanged.
The ONLY executable difference is that:

```python
# v4:
agreement = (judge_z × probe_z > 0)
cap = BASE_CAP + agreement × (MAX_CAP - BASE_CAP)

# v4.1:
cap = sigmoid(abs(probe_z)) × MAX_CAP
```

## 2. The sign test's blind spot

The sign test opens the cap when judge and probe point the same way, and
throttles it to BASE_CAP (2 steps) when they disagree. On Iris, the
disagreement rows are exactly where the probe is right and the judge is
wrong. The gate silences the probe on the rows where it would help most.

This is visible in the v4 results. The probe alone (`sonic_v3_3_mini`) scores
Iris AUROC 0.9918. But v4 delivers only 0.9516 Iris — the gate discards most
of the probe's signal. The Iris Qwen deficit vs Tiresias (-0.0242) is
entirely a fusion problem: the raw probe is 0.9982 on that unit, but the gate
blocks it on disagreement rows.

## 3. How the confidence gate works

The cap is decoupled from the judge. It opens based on probe certainty alone:

| abs(probe_z) | sigmoid | cap | cap (steps) |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.50 | 0.208 | 2.0 |
| 0.5 | 0.62 | 0.259 | 2.5 |
| 1.0 | 0.73 | 0.305 | 2.9 |
| 2.0 | 0.88 | 0.367 | 3.5 |
| 3.0 | 0.95 | 0.397 | 3.8 |
| 4.0 | 0.98 | 0.409 | 3.9 |

- **On Iris**, the probe is strong (AUROC 0.9918, abs(z) routinely > 3). The
  cap opens to near MAX_CAP on most rows, regardless of what the judge says.
  The probe's disagreement corrections get through.

- **On Notus**, the probe is near-random (AUROC 0.5586, abs(z) ≈ 0 on most
  rows). sigmoid(0) = 0.5, so cap = 0.5 × MAX_CAP = 0.208 = **exactly
  BASE_CAP** — the identical 2-step floor as v4's sign test on disagreement
  rows. The Notus floor is unchanged.

The formula is parameter-free. There is no threshold, no agreement check, and
one constant (MAX_CAP). BASE_CAP is no longer used because `sigmoid(0) ×
MAX_CAP = MAX_CAP/2 = BASE_CAP` — the math guarantees the same safety floor.

## 4. Actual results — submitted 2026-07-27

**Headline: AUROC 0.9003, Balanced Accuracy 0.8154.** Runtime 28m 14s.
Regression vs v4: **−0.0057 AUROC, −0.0050 BA**.

| unit | v4 | v4.1 | Δ |
| --- | ---: | ---: | ---: |
| Iris nemotron g-st | 0.9723 | 0.9753 | +0.0030 |
| Iris qwen g-st | 0.9677 | 0.9656 | −0.0021 |
| Iris gemma g-st | 0.9148 | 0.9148 | 0.0000 |
| Notus nemotron | 0.8864 | 0.8484 | **−0.0380** |
| Notus qwen | 0.8427 | 0.8450 | +0.0023 |
| Notus gemma | 0.8525 | 0.8525 | 0.0000 |
| **headline** | **0.9060** | **0.9003** | **−0.0057** |

### Why the confidence gate failed

Notus Nemotron collapsed from 0.8864 to 0.8484 (−0.0380). The v4.1 FPR
rose to 0.500 (random guessing) against v4's 0.370. The probe has enough
spurious signal on Notus that even `sigmoid(|z|)` opens the cap beyond the
safe floor — at |z| = 0.5, the cap is already 2.5 steps (vs v4's 2.0).
On rows where the probe happened to be modestly confident but wrong, the
extra 0.5 steps of influence were enough to damage the ranking.

The local dev sweep got this wrong. Dev is in-sample for the probe, so
|probe_z| is always well-calibrated there — the probe knows when it's
uncertain. On Notus, the probe is out-of-distribution and can be spuriously
confident on the wrong rows. The sweep could not reproduce this regime.

### The sign test was the safety mechanism

v4's sign test opens the cap only when judge and probe *agree on direction*.
This couples the probe's influence to the judge's verdict. On Notus, where
the judge is reasonably good (0.8640) and the probe is random, agreement is
essentially a coin flip — but it's a coin flip *conditioned on the judge's
already-decent ranking*. The probe only amplifies rows where the judge
already points the right way.

The confidence gate decoupled the probe from the judge entirely. On Notus,
it let the probe through on rows where the probe was wrong and the judge
was right — exactly the rows the sign test was designed to block.

## 5. Local dev sweep (L46 only, 21 datasets)

The L40 probe is not cached locally, so the sweep uses L46. Dev is in-sample
for the probe — treat this as a wiring check, not a Notus prediction.

| gate | mean AUROC | worst AUROC |
| --- | ---: | ---: |
| judge alone | 0.9528 | 0.7938 |
| sign (v4) | 0.9832 | 0.9229 |
| sig(abs(z)-2.0) | 0.9689 | 0.8590 |
| sig(abs(z)-1.0) | 0.9786 | 0.9025 |
| sig(abs(z)-0.5) | 0.9827 | 0.9216 |
| sig(abs(z)) no threshold | **0.9857** | **0.9358** |

The no-threshold version is the clear winner: +0.0025 mean over the sign
gate, +0.0129 worst-case. Every threshold makes it worse — thresholds push
the sigmoid right, delaying cap opening without any benefit. The
parameter-free sigmoid is both simpler and strictly better.

## 6. Build

```bash
python experiments/ensemble_gate_eval/build_sonic_v4_1_notebook.py
```

The source is `legacy_submissions/sonic_v4.ipynb`. The build script patches cell 12 only.
The remaining 12 cells are asserted byte-identical to v4.

## 7. Confirmed failure mode

The risk identified in the local sweep was wrong. The probe on Notus *does*
have enough spurious confidence to open the cap beyond the safe floor, and
the extra nudge was enough to cause material ranking damage (−0.0380 on
Notus Nemotron alone). The sign test's coupling to the judge — which the
confidence gate removed — was the safety mechanism.

The local dev sweep predicted +0.0025 mean gain. The actual result was
−0.0057 loss. Dev cannot reproduce Notus dynamics because the probe is
always in-distribution there. Any gate that decouples the probe from the
judge must be tested on real Notus data — local sweeps are blind to this
failure mode.
