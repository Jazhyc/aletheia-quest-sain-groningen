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

- **4-step MAX_CAP may be too conservative or too aggressive.** The v8 judge
  is stronger than v6.2, so the sign test opens the cap more often (larger
  |judge_z| means agreement is more informative). If Iris gains are too small,
  raise MAX_CAP to 5–6 steps. If Notus is degraded, lower to 3 steps.

- **The probe's calibration on Nemotron** (FPR 0.600 in v6, 0.090 in v7.2
  iris rows) is probe-dependent. The gate's sign test should prevent the worst
  damage, but per-family probe calibration offsets can affect BA without
  affecting AUROC.

- **NDIF dual extraction.** Two `.save()` calls in one trace doubles the
  failure surface. The retry logic (MAX_ATTEMPTS=4) handles transient errors.
  Persistent NDIF issues would collapse both probes → judge-only fallback.

## Expected outcome

The v8 judge alone scores headline AUROC 0.9440. The probe's largest
contributions are on Iris Qwen (+0.032 over v8 judge) and Iris Nemotron
(+0.019). At 4-step MAX_CAP, the probe should capture most of this gap on
agreement rows while staying silent on Notus (probe_z ≈ 0). Conservative
projection: headline AUROC ~0.950–0.955.

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
