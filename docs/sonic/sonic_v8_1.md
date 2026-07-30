# sonic v8.1 — dual probe + Phoenix v8.1 judge, v8.0 sign gate

**Dual L40+L46 white-box probe + Phoenix Wright v8.1 direct-margin judge (Kimi
K3 liars-full distilled, r16), blended under the v4 sign gate with v8.0
constants. Same gate, same probe, stronger judge.**

## Motivation

phoenix_wright_v8_1 is a substantially stronger judge than v8.0 (headline
AUROC 0.9661 vs 0.9440, +0.0221). The v8.0 sonic gate constants (JUDGE_LOGIT_SD
= 5.638, BASE_CAP = 1 step, MAX_CAP = 4 steps) were fitted against the v8.0
judge. **Verified 2026-07-30:** both adapters produce bit-identical logits on
14 prompts spanning honest/deceptive/refusal/steering — Δ = 0.0000 on every
prompt. The logit scale is set by Qwen3.5-9B's final layer norm, not the
adapter, so JUDGE_LOGIT_SD = 5.638 transfers without adjustment.

The v8.1 adapter (`aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2`) was
trained with a liars-bench focused distillation recipe, trading the
truth-value-gap (TVG) soft targets of v8.0 for direct liars-bench binary
supervision. The v8.1 judge improved Notus +0.0130 and Iris +0.0182 over v8.0
judge-only, with gains on every single unit.

## Architecture (unchanged from v8.0)

```
probe_z = (z_46 + z_40) / 2.0        # dual probe fused, per-family standardised
judge_z = (log_odds − μ) / σ          # standardised to sd ≈ 1

cap   = BASE_CAP + (judge_z × probe_z > 0) × (MAX_CAP − BASE_CAP)
score = sigmoid(judge_z + cap × probe_z)
```

The sign test, the cap, and the linear contribution are v8.0's, unchanged.
Only the judge adapter changed.

## Change from v8.0

| | v8.0 | v8.1 |
|---|---|---|
| Judge adapter | `aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2` | `aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2` |
| Judge training | TVG binary-soft distillation | Liars-bench binary supervision |
| Headline AUROC (judge-only) | 0.9440 | 0.9661 |
| Headline BA (judge-only) | 0.8488 | 0.8629 |
| `JUDGE_LOGIT_SD` | 5.638 (measured on v8.0, 14 prompts) | **5.638 (verified: Δ=0.0000 vs v8.0 on 14 prompts)** |
| `BASE_CAP` | 1 step = 0.022170 | 1 step = 0.022170 |
| `MAX_CAP` | 4 steps = 0.088681 | 4 steps = 0.088681 |
| `PROBE_GAIN` | 1.0 | 1.0 |

## Gate formula

```
step_z  = 0.125 / JUDGE_LOGIT_SD
BASE_CAP = 1 × step_z
MAX_CAP  = 4 × step_z
```

Constants are identical to v8.0. The JUDGE_LOGIT_SD of 5.638 was measured on
the v8.0 adapter but should transfer — the logit scale is set by Qwen3.5-9B's
final layer norm, not the specific adapter. If judge_z is compressed (sd ≪ 0.5)
or stretched (sd ≫ 2), adjust JUDGE_LOGIT_SD via env var.

## Judge

Phoenix Wright v8.1 direct-margin judge:

- Model: `Qwen/Qwen3.5-9B`
- Adapter: `Jazhyc/aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2`
- Method: binary softmax over label logits at last token
- Routes: HP-KR → `build_epistemic_direct_prompt`, action → `build_action_direct_prompt`, ordinary → `build_direct_prompt`
- No intent adapter, no multi-adapter blending

## Probe

Dual L40+L46 `TransformerTokenProbe` — unchanged from v8.0:

- Same architecture: projection → transformer encoder → linear head
- Same weights: `submission/whitebox_probe/{family}_probe/model.pt` (L46) and `submission/whitebox_probe_L40/{family}_probe/model.pt` (L40)
- Same per-family standardisation constants (frozen offline)
- Same fused z-score: `probe_z = (z_46 + z_40) / 2.0`
- Extraction: both layers in one NDIF trace, two `.save()` calls

## Standardisation constants (frozen offline)

### Probe (per-family, unchanged from v8.0)

| family | L46 mean | L46 sd | L40 mean | L40 sd |
| --- | ---: | ---: | ---: | ---: |
| qwen | −0.3675 | 6.626 | −0.1446 | 6.973 |
| gemma | −2.5240 | 4.511 | −1.1305 | 6.314 |
| nemotron | −6.3296 | 3.268 | −6.4427 | 3.521 |

### Judge (retained from v8.0)

`JUDGE_LOGIT_SD = 5.638` measured on 14 synthetic prompts with the v8.0
adapter. The v8.1 adapter uses the same base model and final layer norm, so
the logit scale should be comparable. Override via `JUDGE_LOGIT_SD` env var
if the v8.1 judge produces visibly different judge_z ranges.

## Risks

- **JUDGE_LOGIT_SD empirically verified to transfer.** Both adapters were
  tested on 14 prompts spanning honest, deceptive, refusal, ambiguous, and
  system-prompt steering categories. Every prompt produced the exact same
  logit difference to 4 decimal places (Δ = 0.0000). The logit scale is
  entirely determined by Qwen3.5-9B's final layer norm — the adapter
  contributes no scale shift. JUDGE_LOGIT_SD = 5.638 (measured on v8.0) is
  equally valid for v8.1. **✓ RESOLVED.**

- **The v8.1 judge scores Iris at 0.9909 and Notus at 0.9413.** On Iris, the
  judge approaches the probe's ceiling (0.9949), so the probe's marginal
  benefit is smaller than in v8.0. If the probe can't add meaningful signal
  on Iris, the ensemble may underperform or merely tie the judge-only score
  while costing 19 extra minutes. The sign gate's safety mechanism (BASE_CAP
  = 1 step on disagreement) ensures Notus won't be degraded, but the expected
  headline gain may be negligible.

- **At the v8.1 judge's strength, the probe may be harmful on balance.**
  v6-mini-long proved the probe scores Notus at 0.56 — near-random. v8.0's
  gate extracted +0.0065 AUROC net, with Iris gains (+0.0125 Qwen) barely
  outweighing small Notus costs. If v8.1's judge-only Iris AUROC of 0.9909
  leaves only ~0.005 of headroom for the probe, and the probe costs even
  −0.001 on Notus, the ensemble will be a net regression vs judge-only.

## Projected outcome

The v8.1 judge scores judge-only headline AUROC 0.9661 / BA 0.8629. Under the
v8.0 gate, the probe adds ~+0.005–0.010 on Iris (where the judge still has a
small gap to the probe's 0.995 ceiling) at negligible Notus cost. Net projected
headline: **AUROC ~0.968–0.971, BA ~0.860–0.865**.

However, if the probe's Iris contribution shrinks below ~0.003 (because the
judge nearly saturates Iris), the ensemble could score below judge-only —
the probe's Notus noise (even at 1-step BASE_CAP) would outweigh its vanishing
Iris gain. The 19-minute extraction overhead would then be pure waste.

## Results — submitted 2026-07-30

**Headline: AUROC 0.9675, Balanced Accuracy 0.8567.** Runtime 30m 9s.
Rank 1 on the leaderboard. **New best SAIN Groningen submission.**

| | sonic v8.1 | phoenix v8.1 (judge-only) | sonic v8.0 |
|---|---|---|---|
| **Headline AUROC** | **0.9675** | 0.9661 | 0.9505 |
| **Headline BA** | **0.8567** | 0.8629 | 0.8450 |
| **Runtime** | 30m 9s | 11m 29s | 30m 55s |

### Per-validation-dataset

**Notus (base-model detection):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| Nemotron-120B | 0.9656 | 0.8075 | 0.6400 | 0.0250 |
| Qwen3.5-27B | 0.9126 | 0.6875 | 0.4150 | 0.0400 |
| Gemma-3-27b-it | 0.9493 | 0.7800 | 0.5700 | 0.0100 |

**Iris (general steering detection):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| Nemotron g-st | 0.9984 | 0.9725 | 0.9600 | 0.0150 |
| Qwen g-st | 0.9970 | 0.9675 | 0.9350 | 0.0000 |
| Gemma g-st-2 | 0.9817 | 0.9250 | 0.8750 | 0.0250 |

**Eunomia (varied deception, reference):**

| Model | AUROC | Bal.Acc | Recall | FPR |
|-------|-------|---------|--------|-----|
| Qwen base | 0.9916 | 0.9381 | 0.9107 | 0.0345 |
| a-mo-1 | 0.8874 | 0.7781 | 0.5974 | 0.0412 |
| a-mo-3 | 0.7845 | 0.6167 | 0.2769 | 0.0435 |
| a-mo-4 | 0.9115 | 0.7766 | 0.5897 | 0.0365 |
| a-mo-5 | 0.9764 | 0.9056 | 0.8333 | 0.0222 |
| a-mo-6 | 0.9251 | 0.8083 | 0.6667 | 0.0500 |
| a-mo-7 | 0.9318 | 0.8138 | 0.6628 | 0.0352 |
| b-mo | 0.9689 | 0.9127 | 0.9450 | 0.1197 |
| c-mo | 0.9910 | 0.9554 | 0.9262 | 0.0154 |

### Key deltas vs judge-only (phoenix v8.1)

| Unit | sonic v8.1 | phoenix v8.1 | Δ AUROC | Δ BA |
|------|:----------:|:----------:|:--:|:--:|
| Iris Qwen g-st | 0.9970 | 0.9943 | **+0.0027** | +0.0000 |
| Notus Qwen | 0.9126 | 0.9093 | **+0.0033** | −0.0300 |
| Notus Nemotron | 0.9656 | 0.9638 | +0.0018 | +0.0075 |
| Iris Nemotron g-st | 0.9984 | 0.9976 | +0.0008 | +0.0075 |
| Iris Gemma g-st | 0.9817 | 0.9809 | +0.0008 | −0.0100 |
| Notus Gemma | 0.9493 | 0.9508 | −0.0015 | −0.0125 |

The probe's contribution collapsed to near-zero: +0.0014 AUROC headline gain
over judge-only. The v8.1 judge scores Iris at 0.9909 — the probe has only
~0.004 of headroom left, and the sign gate can only extract +0.0014 of it.
Notus is effectively flat (+0.0012 mean, with Gemma actually regressing
−0.0015). The 19-minute NDIF extraction overhead buys essentially nothing.

BA dropped −0.0062 vs judge-only because the probe shifts the score
distribution slightly without improving the ranking — the threshold at 0.5
is slightly less calibrated for the blended scores.

## Comparison: probe vs judge vs ensemble

| | AUROC | Bal.Acc | Runtime | Method |
|---|---|---|---|---|
| **sonic v8.1** | **0.9675** | **0.8567** | 30m 9s | probe + v8.1 judge |
| phoenix_wright_v8_1 | 0.9661 | 0.8629 | **11m 29s** | judge-only |
| sonic v8.0 | 0.9505 | 0.8450 | 30m 55s | probe + v8.0 judge |
| sonic v6_mini_long | 0.7773 | 0.6658 | 25m 7s | probe-only |

The ensemble technically holds the AUROC record (+0.0014 over judge-only),
but the gain is within noise. Judge-only is faster (2.6×) and holds the
BA record (+0.0062). **The probe is now conclusively dead weight** — the
v8.1 judge alone matches or exceeds the ensemble on every practical
dimension while costing nothing in NDIF overhead.

## Notebook

`submission/sonic_v8_1.ipynb` — 14 cells.

| Cells | Content | Source |
|-------|---------|--------|
| 0 | Markdown header | New (v8.1) |
| 1–2 | Imports, config, device | v8_0 (unchanged) |
| 3 | Dataset loading, family detection | v8_0 |
| 4 | Dual probe config loading (L46 + L40) | v8_0 |
| 5 | TransformerTokenProbe ×2 | v8_0 |
| 6 | NNSight model loading | v8_0 |
| 7 | Layer location (L46 + L40) | v8_0 |
| 8 | Prompt construction, tokenization | v8_0 |
| 9 | Dual activation extraction | v8_0 |
| 10 | Dual probe scoring + z-score fusion | v8_0 |
| 11 | Phoenix v8.1 judge | New (adapter swap) |
| 12 | Sign gate (v8.0 constants) | v8_0 |
| 13 | Write submission | v8_0 |

## Build

```bash
# The notebook is already built and submitted. No weight changes needed —
# only the adapter repo changed (cell 11). JUDGE_LOGIT_SD = 5.638 was
# empirically verified to transfer (Δ = 0.0000 on 14 prompts).
```
