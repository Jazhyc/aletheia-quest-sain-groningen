# sonic v8.0 — dual probe + Phoenix v8 judge, retuned v4 sign gate

**Dual L40+L46 white-box probe + Phoenix Wright v8.0 direct-margin judge (Kimi
K3 distilled, r16), blended under the v4 sign gate with retuned constants for
the v8 judge.**

## Motivation

v6.1 used the v6.2 judge (multi-adapter, intent blending, `JUDGE_LOGIT_SD =
6.0`). Phoenix v8 is a substantially stronger judge (headline AUROC 0.9440 vs
v6.2's ~0.9233). The sign gate constants must be recomputed for the v8 judge's
score distribution.

The v8 judge outputs `softmax(label_logits)[:, 1]` — a probability in [0,1],
not a raw logit margin. The log-odds are recovered via the logit transform,
then standardised to mean≈0, sd≈1 to match the probe_z scale.

## Architecture (unchanged from v4/v6)

```
probe_z = (z_46 + z_40) / 2.0        # dual probe fused, per-family standardised
judge_z = (log_odds − μ) / σ          # standardised to sd ≈ 1

cap   = BASE_CAP + (judge_z × probe_z > 0) × (MAX_CAP − BASE_CAP)
score = sigmoid(judge_z + cap × probe_z)
```

The sign test, the cap, and the linear contribution are v4's, unchanged. The
only change is the numeric values of `BASE_CAP`, `MAX_CAP`, and the judge
standardisation constants.

## Change from v6.1

| | v6.1 | v8.0 |
|---|---|---|
| Judge | Phoenix v6.2 (multi-adapter, intent) | Phoenix v8.0 (Kimi K3, single adapter) |
| Judge output | raw logit margin | softmax probability → log-odds |
| `JUDGE_SD` | 6.0 (normalisation divisor) | **5.638 (measured, see below)** |
| 1 bf16 step in z | 0.125 / 6.0 = 0.020833 | **0.125 / 5.638 = 0.022170** |
| `BASE_CAP` | 1 step = 0.020833 | **1 step = 0.022170** |
| `MAX_CAP` | 3 steps = 0.062500 | **4 steps = 0.088681** |
| `PROBE_GAIN` | 1.0 | 1.0 |

`MAX_CAP` raised from 3 to 4 steps. The v8 judge is stronger (0.944 vs 0.923),
so a slightly larger cap can safely let through more probe signal on Iris
without endangering Notus.

`BASE_CAP` kept at 1 step. v3.3's evidence: 1-step BASE_CAP scored the best
Notus mean of any sonic run. On disagreement rows (where the probe may be
wrong), even 1 step is enough to break ties but not to reorder the ranking.

## Gate formula

```
step_z  = 0.125 / JUDGE_LOGIT_SD
BASE_CAP = 1 × step_z
MAX_CAP  = 4 × step_z
```

### Measuring JUDGE_LOGIT_SD

Measured on 14 synthetic prompts (clean, deceptive, refusal, ambiguous,
system-prompt steering) on a local RTX 4090 with Qwen3.5-9B + v8 adapter.

| | value |
|---|---|
| N samples | 14 |
| logit_diff sd | 5.638 |
| logit_diff range | [−6.6, +7.6] |
| derived step_z | 0.022170 |
| derived BASE_CAP (1 step) | 0.022170 |
| derived MAX_CAP (4 steps) | 0.088681 |

The sd is close to v6.2's 6.0 — the logit scale is set by Qwen3.5-9B's final
layer norm, not the specific adapter. Override via `JUDGE_LOGIT_SD` env var.

## Judge

Phoenix Wright v8.0 direct-margin judge:

- Model: `Qwen/Qwen3.5-9B`
- Adapter: `Jazhyc/aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2`
- Method: binary softmax over label logits at last token
- Routes: HP-KR → `build_epistemic_direct_prompt`, action → `build_action_direct_prompt`, ordinary → `build_direct_prompt`
- No intent adapter, no multi-adapter blending

## Probe

Dual L40+L46 `TransformerTokenProbe` — unchanged from v6.1:

- Same architecture: projection → transformer encoder → linear head
- Same weights: `submission/whitebox_probe/{family}_probe/model.pt` (L46) and `submission/whitebox_probe_L40/{family}_probe/model.pt` (L40)
- Same per-family standardisation constants (frozen offline)
- Same fused z-score: `probe_z = (z_46 + z_40) / 2.0`
- Extraction: both layers in one NDIF trace, two `.save()` calls

## Standardisation constants (frozen offline)

### Probe (per-family, unchanged from v6.1)

| family | L46 mean | L46 sd | L40 mean | L40 sd |
| --- | ---: | ---: | ---: | ---: |
| qwen | −0.3675 | 6.626 | −0.1446 | 6.973 |
| gemma | −2.5240 | 4.511 | −1.1305 | 6.314 |
| nemotron | −6.3296 | 3.268 | −6.4427 | 3.521 |

### Judge (to be measured from v8 on dev data)

`JUDGE_LOGIT_MEAN` and `JUDGE_LOGIT_SD` must be measured by running the v8
judge on the full dev set and computing the mean and standard deviation of
`log(p / (1-p))` across all rows.

## Risks

- **`JUDGE_LOGIT_SD = 5.638` was measured on 14 synthetic prompts, not the
  full dev set.** The true SD on the Aletheia dev data may differ slightly.
  The notebook prints `judge_z` range at runtime — verify it spans roughly
  [−2, +2] (sd ≈ 0.5–1.0). If judge_z is compressed (sd ≪ 1), raise
  `JUDGE_LOGIT_SD`. If stretched (sd ≫ 1), lower it.
  **Post-submission: ✓ RESOLVED.** The 5.638 constant proved accurate — Notus
  is flat (±0.008 AUROC, no collapse) and Iris gains are real (+0.01–0.013).
  The SD measurement on synthetic prompts matched is good enough.

- **4-step MAX_CAP may be too conservative or too aggressive.** The v8 judge
  is stronger than v6.2, so the sign test opens the cap more often (larger
  |judge_z| means agreement is more informative). If Iris gains are too small,
  raise MAX_CAP to 5–6 steps. If Notus is degraded, lower to 3 steps.
  **Post-submission: ✓ RESOLVED.** 4 steps is well-calibrated. Iris gained
  +0.0125 (Qwen) and +0.0101 (Nemotron). Notus is not degraded (actually
  improved +0.0049 mean). No major adjustment needed.

- **The probe's calibration on Nemotron** (FPR 0.600 in v6, 0.090 in v7.2
  iris rows) is probe-dependent. The gate's sign test should prevent the worst
  damage, but per-family probe calibration offsets can affect BA without
  affecting AUROC.
  **Post-submission: ✓ CONFIRMED, MINIMAL IMPACT.** BA dropped from 0.8488 to
  0.8450 (−0.0038). Notus Nemotron FPR 0.055 vs judge-only 0.050 — a tiny
  threshold shift. Iris Nemotron FPR actually improved (0.085 vs 0.090). The
  calibration offset is real but harmless at this magnitude.


## Results — submitted 2026-07-30

**Headline: AUROC 0.9505, Balanced Accuracy 0.8450.** Runtime 30m 55s.

| | sonic v8.0 | phoenix v8 (judge-only) | sonic v7.2 (organism router) |
|---|---|---|---|
| **Headline AUROC** | **0.9505** | 0.9440 | 0.9251 |
| **Headline BA** | **0.8450** | 0.8488 | 0.8371 |
| **Runtime** | 30m 55s | 11m 47s | 29m 38s |

### Per-validation-dataset

**Notus (base-model detection):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| nvidia/Nemotron-3-Super-120B | 0.9522 | 0.8400 | 0.7350 | 0.0550 |
| Qwen/Qwen3.5-27B | 0.8947 | 0.7200 | 0.5000 | 0.0600 |
| google/gemma-3-27b-it | 0.9379 | 0.8025 | 0.6500 | 0.0450 |

**Iris (general steering detection):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| Nemotron g-st | 0.9812 | 0.9125 | 0.9100 | 0.0850 |
| Qwen g-st | 0.9791 | 0.9150 | 0.8650 | 0.0350 |
| Gemma g-st-2 | 0.9577 | 0.8800 | 0.8300 | 0.0700 |

**Eunomia (varied deception, reference):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| Qwen base | 0.9917 | 0.9450 | 0.9107 | 0.0207 |
| a-mo-1 | 0.8876 | 0.7509 | 0.5584 | 0.0567 |
| a-mo-3 | 0.7933 | 0.6249 | 0.2769 | 0.0272 |
| a-mo-4 | 0.9071 | 0.7766 | 0.5897 | 0.0365 |
| a-mo-5 | 0.9796 | 0.8983 | 0.8077 | 0.0111 |
| a-mo-6 | 0.9236 | 0.8119 | 0.6667 | 0.0429 |
| a-mo-7 | 0.9321 | 0.8126 | 0.6744 | 0.0493 |
| b-mo | 0.9663 | 0.9109 | 0.9500 | 0.1282 |
| c-mo | 0.9862 | 0.9477 | 0.9262 | 0.0308 |

### Key deltas vs judge-only (phoenix v8)

| Unit | sonic v8.0 | phoenix v8 | Δ |
|------|:----------:|:----------:|:--:|
| Iris Qwen g-st | 0.9791 | 0.9666 | **+0.0125** |
| Iris Nemotron g-st | 0.9812 | 0.9711 | **+0.0101** |
| Iris Gemma g-st | 0.9577 | 0.9556 | +0.0021 |
| Notus Nemotron | 0.9522 | 0.9444 | +0.0078 |
| Notus Qwen | 0.8947 | 0.8873 | +0.0074 |
| Notus Gemma | 0.9379 | 0.9386 | −0.0007 |

The probe delivers its largest gains on Iris (+0.0125 Qwen, +0.0101 Nemotron)
at negligible Notus cost (−0.0007 Gemma, +0.0074–0.0078 on the other two).
Net headline gain over judge-only: **+0.0065 AUROC**.

## Build

```bash
# Step 1: copy L40 weights (done)
cp -r legacy_submissions/whitebox_probe_L40 submission/whitebox_probe_L40/

# Step 2: build notebook (done)
# Cells 1–10 byte-identical to sonic_v6_1
# Cell 11: phoenix v8 judge (new)
# Cell 12: retuned sign gate (new)

# Step 3: measure JUDGE_LOGIT_SD from v8 judge on dev data (TODO)
# Set JUDGE_LOGIT_SD env var to empirical value before submission
```

## Notebook

`submission/sonic_v8_0.ipynb` — 14 cells.

| Cells | Content | Source |
|-------|---------|--------|
| 0 | Markdown header | New |
| 1–2 | Imports, config, device | v6_1 (unchanged) |
| 3 | Dataset loading, family detection | v6_1 |
| 4 | Dual probe config loading (L46 + L40) | v6_1 |
| 5 | TransformerTokenProbe ×2 | v6_1 |
| 6 | NNSight model loading | v6_1 |
| 7 | Layer location (L46 + L40) | v6_1 |
| 8 | Prompt construction, tokenization | v6_1 |
| 9 | Dual activation extraction | v6_1 |
| 10 | Dual probe scoring + z-score fusion | v6_1 |
| 11 | Phoenix v8 judge | New |
| 12 | Sign gate (retuned) | New |
| 13 | Write submission | v6_1 |
