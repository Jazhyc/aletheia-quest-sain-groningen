# sonic v6 — dual probe + Phoenix v6.2 direct-margin judge under the v4 sign gate

**Dual L40+L46 white-box probe + Phoenix Wright v6.2 direct-label-logit judge,
blended under the v4 sign gate (no escalation).**

## Motivation

v6 combines the two strongest components available:

1. **Dual probe (L40+L46)** — from v6-mini-long.  Fused as
   `probe_z = (z_46 + z_40) / 2.0`, frozen offline constants per family.
   No token cap.

2. **Phoenix v6.2 direct-margin judge** — binary-soft-distilled rank-16 adapter
   on Qwen3.5-9B, with HP-KR, action, and reasoning-intent specialists.
   `score_messages()` from the `phoenix_wright` module.

3. **v4 sign gate** — probe influence bounded at `BASE_CAP` (2 steps) when
   judge and probe disagree, opens to `MAX_CAP` (4 steps) when they agree:

```
score = sigmoid(judge_z + cap × probe_z)
cap   = BASE_CAP + (judge_z · probe_z > 0) × (MAX_CAP − BASE_CAP)
```


## Change from v6-mini-long

| | v6-mini-long | v6 |
|---|---|---|
| Probe | dual L40+L46, max_len=0 | same |
| Judge | none | Phoenix v6.2 (multi-route direct-margin) |
| Scoring | `sigmoid(probe_z)` | sign gate: `sigmoid(judge_z + cap · probe_z)` |
| Cells | 12 | 14 |

## Expected outcome

If the v6.2 judge's Notus Qwen weakness (AUROC 0.8550) is the bottleneck, the
probe can't help there (it's also dead on Notus Qwen).  But on Iris — where the
probe transfers strongly and the v6.2 judge scores 0.9574 — the gate should
nudge above the judge-alone score.  On Metis and Eunomia the probe dominates and
should pull the blended score above both components.


## Results

Submitted Jul 29, 2026 · 10:36 AM.  Headline AUROC **0.9230** / BA **0.8013**,
runtime 35m 32s.

| | AUROC | Bal.Acc | Runtime |
|---|---|---|---|
| **Headline (Notus + Iris)** | **0.9230** | **0.8013** | 35m 32s |

### Per-dataset

**Notus (validation):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| nvidia/Nemotron-3-Super-120B | 0.8224 | 0.6850 | 0.9700 | 0.6000 |
| Qwen/Qwen3.5-27B | 0.8725 | 0.6875 | 0.4100 | 0.0350 |
| google/gemma-3-27b-it | 0.9019 | 0.7325 | 0.4800 | 0.0150 |

**Iris (validation):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| nvidia/Nemotron-3-Super-120B (adapter) | 0.9893 | 0.9250 | 0.9700 | 0.1200 |
| Qwen/Qwen3.5-27B (adapter) | 0.9944 | 0.9625 | 0.9400 | 0.0150 |
| google/gemma-3-27b-it (adapter) | 0.9577 | 0.8150 | 0.6500 | 0.0200 |

**Metis (reference only):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| nvidia/Nemotron-3-Super-120B | 1.0000 | 0.9875 | 1.0000 | 0.0250 |
| Qwen/Qwen3.5-27B | 0.9998 | 0.9900 | 0.9900 | 0.0100 |
| Qwen3.5-27B a-mo-1 | 0.9993 | 0.9830 | 0.9709 | 0.0050 |
| Qwen3.5-27B a-mo-3 | 0.9991 | 0.9850 | 0.9850 | 0.0150 |
| Qwen3.5-27B a-mo-4 | 0.9992 | 0.9900 | 0.9900 | 0.0100 |
| Qwen3.5-27B a-mo-5 | 0.9985 | 0.9900 | 0.9850 | 0.0050 |
| Qwen3.5-27B a-mo-6 | 0.9998 | 0.9894 | 0.9838 | 0.0050 |
| Qwen3.5-27B a-mo-7 | 0.9993 | 0.9893 | 0.9837 | 0.0050 |
| Qwen3.5-27B b-mo | 0.9995 | 0.9863 | 0.9925 | 0.0200 |
| Qwen3.5-27B c-mo | 0.9991 | 0.9800 | 0.9950 | 0.0350 |
| Qwen3.5-27B g-st | 0.9995 | 0.9875 | 1.0000 | 0.0250 |
| google/gemma-3-27b-it | 0.9945 | 0.9750 | 1.0000 | 0.0500 |
| gemma g-st-gemma-3-27b-it-2 | 0.9804 | 0.9500 | 1.0000 | 0.1000 |
| gemma s-mo-gemma-3-27b-it | 0.9962 | 0.9850 | 0.9950 | 0.0250 |

**Eunomia (reference only):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| Qwen/Qwen3.5-27B | 0.9980 | 0.9742 | 0.9554 | 0.0069 |
| Qwen3.5-27B a-mo-1 | 0.9013 | 0.7534 | 0.5584 | 0.0515 |
| Qwen3.5-27B a-mo-3 | 0.8288 | 0.6452 | 0.3231 | 0.0326 |
| Qwen3.5-27B a-mo-4 | 0.9152 | 0.8023 | 0.6410 | 0.0365 |
| Qwen3.5-27B a-mo-5 | 0.9744 | 0.8936 | 0.8205 | 0.0333 |
| Qwen3.5-27B a-mo-6 | 0.9362 | 0.8155 | 0.6667 | 0.0357 |
| Qwen3.5-27B a-mo-7 | 0.9391 | 0.8441 | 0.7093 | 0.0211 |
| Qwen3.5-27B b-mo | 0.9827 | 0.9177 | 0.9550 | 0.1197 |
| Qwen3.5-27B c-mo | 0.9912 | 0.9521 | 0.9426 | 0.0385 |

## Findings

**⚠️ Post-mortem (Jul 29): normalisation mismatch discovered.** The v4 gate
constants (`BASE_CAP = 0.20838`, `MAX_CAP = 0.41675`) were calibrated for
`JUDGE_MARGIN_SD ≈ 1.19976` but the v6.2 judge uses `JUDGE_LOGIT_SD = 6.0`.
A bf16 step in standardised space is `0.125 / JUDGE_SD`, so the old values
represent **10 steps** (`BASE_CAP`) and **20 steps** (`MAX_CAP`) in the v6.2
system — 5–10× the intended probe influence. This is the root cause of the
Notus Nemotron collapse. See `docs/sonic/sonic_v6_1.md` for the fix.

**v6 does not beat the judge alone.** The headline AUROC 0.9230 is −0.0003
below phoenix_wright_v6.2 (0.9233) and BA drops from 0.8379 to 0.8013
(−0.0366). The ensemble adds ~19 minutes of runtime for a net loss.

**The probe helps Iris, hurts Notus — and they nearly cancel.**

| | pw_v6.2 (judge-only) | v6 (ensemble) | Δ v6 vs pw |
|---|---|---|---|
| Notus Nemotron | 0.9142 | 0.8224 | **−0.0918** |
| Notus Qwen | 0.8550 | 0.8725 | +0.0175 |
| Notus Gemma | 0.8983 | 0.9019 | +0.0036 |
| Iris Nemotron | 0.9643 | 0.9893 | +0.0250 |
| Iris Qwen | 0.9526 | 0.9944 | **+0.0418** |
| Iris Gemma | 0.9554 | 0.9577 | +0.0023 |
| **Notus mean** | 0.8892 | 0.8656 | −0.0236 |
| **Iris mean** | 0.9574 | 0.9805 | +0.0230 |
| **headline** | **0.9233** | **0.9230** | **−0.0003** |

The Iris gains (+0.0230 mean) are real and track the probe's transfer profile —
Iris Qwen +0.0418 is the largest single-unit gain of any ensemble run since
v3.1. But Notus Nemotron collapses from 0.9142 to 0.8224 (−0.0918), eating the
entire Iris gain. Notus Qwen actually improves (+0.0175), so the damage is
family-specific, not a uniform Notus problem.

**The sign gate is not safe on Nemotron.** v4's sign test was designed to keep
the probe muted on Notus by keeping the cap at BASE_CAP when judge and probe
disagree. But the v6.2 judge scores Notus Nemotron at 0.9142 AUROC — it's very
strong there. The probe at 0.6088 is weak but above chance. When a strong judge
and a weak probe agree on direction (which happens ~55% of the time at AUROC
0.56), the cap opens to MAX_CAP and the weak probe gets 4 steps of influence.
On the 45% of rows where they disagree, the probe is locked at 2 steps. The
asymmetric cap amplifies the probe when it's correlated with the judge's
direction but the correlation is so weak that the net effect is damage.

**The v4 gate was tuned for a different judge.** The v4 gate's constants
(BASE_CAP 2 steps, MAX_CAP 4 steps) were fitted against the v4.0 judge
(adapter not loaded, AUROC 0.8640 on Notus). The v6.2 judge scores Notus at
0.8892 — nearly 0.03 higher. The stronger judge opens the cap more often
(because `judge_z × probe_z > 0` is more likely when |judge_z| is larger),
giving the weak probe more room to do damage. The gate needs retuning for the
new judge.

**BA regresses sharply.** v6 BA 0.8013 vs pw_v6.2 0.8379 (−0.0366). The
probe's score distribution is shifted relative to the judge's, and the blend
moves the effective threshold. This is a calibration problem: the sign gate
optimizes ranking (AUROC) but the probe's per-family offset distorts the
decision boundary.

**Next.** The probe helps on Iris and hurts on Notus — the two effects nearly
cancel, leaving v6 as a more expensive but not better version of pw_v6.2. Three
paths forward:

1. **Retune the gate for the v6.2 judge — done in v6.1.** The root cause was
   a normalisation mismatch: v4 constants (calibrated for JUDGE_MARGIN_SD≈1.2)
   applied to a judge using JUDGE_LOGIT_SD=6.0, giving 10–20 steps of probe
   influence instead of 2–4. v6.1 recomputes BASE_CAP = 1 step (0.02083) and
   MAX_CAP = 3 steps (0.0625). See `docs/sonic/sonic_v6_1.md`.
2. **Selective escalation: probe-only on Iris, judge-only on Notus.** The
   per-family quality spread is wide enough (probe AUROC 0.99 Iris, 0.56 Notus)
   that a single gate constant cannot price both. A family-aware gate that
   opens the cap on Iris and locks it shut on Notus could recover the Iris
   gains without paying the Notus penalty.
3. **Fix the probe's Notus transfer.** No gate constant will help if the probe
   is net-negative on Notus. A probe that transfers to Notus at AUROC ~0.75+
   would turn the ensemble net-positive.

## Gate constants

Same fitted values as v4/v5:
- `BASE_CAP = 0.20837585277130496` (2 judge quantization steps)
- `MAX_CAP = 0.41675170554260993` (4 steps)
- `PROBE_GAIN = 1.0` (frozen, pre-standardised probe_z)
- Judge standardisation via frozen `JUDGE_MARGIN_SD` (log-odds space)
