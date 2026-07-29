# sonic v6.1 — dual probe + v6.2 judge, retuned v4 sign gate

**Same architecture as v6 (dual L40+L46 probe + Phoenix v6.2 judge under v4 sign
gate), with retuned constants for the stronger judge.**

## Bug discovered in v6 — normalisation mismatch

v6 used the v4 gate constants (`BASE_CAP = 0.20838`, `MAX_CAP = 0.41675`)
unchanged. These were calibrated against the v4.0 judge, which standardised
its margins with `JUDGE_MARGIN_SD ≈ 1.19976`. The v6.2 judge uses
`JUDGE_LOGIT_SD = 6.0` instead.

A bf16 quantisation step is `0.125` in raw logit/log-odds space. The step
size in standardised judge_z space depends on the normalisation divisor:

```
step_z = 0.125 / JUDGE_SD
```

| system | JUDGE_SD | step_z | BASE_CAP value | effective steps | MAX_CAP value | effective steps |
|---|---|---|---|---|---|---|
| v4 gate × v4 judge | 1.19976 | 0.10419 | 0.20838 | **2** (correct) | 0.41675 | **4** (correct) |
| v4 gate × v6.2 judge | 6.0 | 0.02083 | 0.20838 | **10** (bug) | 0.41675 | **20** (bug) |

The old cap values give the probe **5–10× the intended influence** when the
normalisation divisor is 5× larger. v6 was running with 10–20 steps of probe
nudge instead of the intended 2–4. This explains the Nemotron collapse
(−0.0918 AUROC, FPR 0.600): the probe was effectively dominating the ensemble
on Notus rows where the cap opened.

The correct derivation for the v6.2 judge:

```
bf16_step   = 0.125
JUDGE_SD    = 6.0              # from judge cell: judge_z = log_odds / 6.0
step_z      = 0.125 / 6.0      # = 0.020833...
BASE_CAP    = 1 × step_z        # = 0.020833  (1 step)
MAX_CAP     = 3 × step_z        # = 0.062500  (3 steps)
```

## Change from v6

| constant | v6 (miscalibrated) | v6.1 (retuned) |
|---|---|---|
| `JUDGE_SD` | 1.19976 (from v4, mismatched) | 6.0 (from judge cell) |
| 1 bf16 step in z | 0.10419 | **0.020833** |
| `BASE_CAP` | 0.20838 (10 steps) | **0.020833 (1 step)** |
| `MAX_CAP` | 0.41675 (20 steps) | **0.062500 (3 steps)** |
| `PROBE_GAIN` | 1.0 | 1.0 |

`BASE_CAP` = 1 step follows v3.3's evidence: 1-step `BASE_CAP` scored the best
Notus mean of any sonic run (0.8642). At 1 step, the probe can break ties
within the judge's quantisation but cannot overturn confident judge rankings.

`MAX_CAP` = 3 steps is reduced from v6's effective 20 steps (and from v4's
intended 4). On agreement rows where the stronger v6.2 judge has larger
|judge_z|, even 3 steps gives meaningful Iris nudge without the Nemotron
damage seen at 20 steps.

## Gate formula (unchanged architecture)

```
score  = sigmoid(judge_z + cap × probe_z)
cap    = BASE_CAP + (judge_z × probe_z > 0) × (MAX_CAP − BASE_CAP)
```

The sign test is unchanged: cap opens to `MAX_CAP` only when judge and probe
agree on direction. On disagreement rows, the probe is locked at `BASE_CAP`
(1 step). The entire change is the numeric values of `BASE_CAP` and `MAX_CAP`,
recomputed for the v6.2 judge's normalisation.

## Expected outcome

1. **Notus safety should be restored.** At 1-step `BASE_CAP` and 3-step
   `MAX_CAP`, the probe contributes ~1–3% of the judge's spread — barely enough
   to break ties, not enough to reorder the ranking. The Nemotron collapse
   (−0.0918 at 10–20 steps) should reverse entirely. Notus should return to
   near the judge's standalone 0.8892.

2. **Iris gains should partially survive.** The probe's Iris signal is strong
   (AUROC 0.9949 alone). Even at 3-step `MAX_CAP`, the probe pushes
   high-confidence agreement rows in the right direction. The Iris Qwen gain
   (+0.0418 in v6 at 20 steps) will reduce but should not vanish — a
   perfect-ranking probe doesn't need much weight to break ties.

3. **Headline should exceed judge-only.** Conservative projection: Notus ~0.88
   (near judge's 0.8892), Iris ~0.965 (between judge's 0.9574 and v6's 0.9805).
   Headline ~0.922–0.927, beating the judge's 0.9233.

## Risks

- **`JUDGE_LOGIT_SD = 6.0` is a default, not an empirically fitted value.**
  The judge cell allows override via env var. If the judge's actual log-odds SD
  differs substantially from 6.0, `judge_z` won't have sd≈1 and the effective
  probe influence will differ from the intended step count. The notebook
  includes a print of the judge_z range at runtime — verify it spans roughly
  [−2, +2] (sd≈0.5–1.0) before trusting the step derivation. If judge_z is
  compressed (sd≪1), the cap values are too small and Iris gain vanishes. If
  judge_z is stretched (sd≫1), the cap values are too large and Notus damage
  persists.

- **3-step `MAX_CAP` may be too conservative for Iris.** If Iris gains
  collapse (ensemble Iris indistinguishable from judge-only), `MAX_CAP` can be
  raised to 4–6 steps in v6.2 independently of `BASE_CAP`.

- **The probe's per-family calibration offset on Nemotron** (FPR 0.600 in v6)
  may persist even at 1-step `BASE_CAP`. The probe's z-scores run
  systematically high on Nemotron (v6-mini-long Nemotron probe mean was shifted).
  Even 1 step of influence could push the decision boundary enough to degrade
  BA without affecting AUROC. This is a probe calibration problem, not a gate
  problem — per-family thresholding in the scoring cell would address it.

- **If the original v4 JUDGE_MARGIN_SD (1.19976) was itself wrong** — i.e., if
  it was an empirical SD and the v6.2 judge's log-odds happen to have a very
  different empirical SD — then even the 10×-too-large v6 caps could have been
  accidentally correct. This is unlikely (the v6 results show clear probe
  dominance) but the only way to verify is with one run.

## Next

- **v6.1 (this run)**: 1-step BASE_CAP, 3-step MAX_CAP.
- **v6.2**: if Notus safe but Iris collapsed, raise MAX_CAP to 4–6 steps.
- **v6.3**: if Notus still damaged, try per-family BASE_CAP (1 step on
  Nemotron/gemma, 2 steps on Qwen where probe AUROC is 0.63).
