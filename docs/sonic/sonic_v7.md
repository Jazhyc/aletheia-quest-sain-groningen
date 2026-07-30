# sonic v7.2 — L46 probe (deception + organism heads) routed against Phoenix v8 judge

**Judge:** Phoenix v8 — Qwen3.5-9B + `Jazhyc/aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2`
(Kimi K3 full binary-soft-distilled r16 adapter). Multi-route with
HP-KR, action, and intent specialist judges. Outputs a probability ∈ [0,1] via
`score_messages()` — no logit standardisation.

**Probe:** Single L46 transformer probe with two heads sharing one encoder:
one reads deception, one detects whether the model is an organism. Both output
raw sigmoid probabilities ∈ [0,1] from BCE training. When the organism head
fires, the deception head takes over. Otherwise, the judge handles the row. No
blending, no z-score standardisation.

**Organism head:** Per-family LogisticRegression heads (Linear(128→1)) with
StandardScaler, trained on current encoder's pooled features. Qwen AUROC 0.93,
Gemma AUROC 0.61 (5-fold CV). Scaler folded into Linear weights at load time
so forward pass is a single matmul. Threshold t=0.7. Non-Qwen/Gemma families
route directly to judge (no shared organism head fallback).

**Notebook:** `submission/sonic_v7_2.ipynb`.

### v7 (failed first attempt)

The original sonic v7 (submitted Jul 29 21:06) used a shared organism head without
StandardScaler — the head was trained on scaled features but deployed on raw
features. This deployment bug caused all rows to route to the judge, producing
output byte-identical to the Phoenix v6.3 judge-only submission (headline 0.9303).
v7.1 fixes this with per-family heads, scaler folding, and raised threshold
(t=0.7) but there was a bug in the notebook that caused the probe to be skipped.

## Motivation

Every sonic version since v3.1 faces the same fundamental trade-off:

| dataset type | probe-alone AUROC | judge-alone AUROC | best strategy |
|---|---|---|---|
| Organisms (Iris, Metis, Eunomia) | 0.94–0.99 | 0.85–0.96 | **use the probe** |
| Base models (Notus) | 0.44–0.63 | 0.86–0.91 | **use the judge** |

A static gate constant cannot price both regimes. The sign gate opens the cap
on agreement rows — but on Notus, a strong judge agrees with the weak probe
by chance (~55% of rows at AUROC 0.56), opening the cap where the probe is
wrong. On Iris, the cap constrains a probe that would score 0.99 alone.

The organism head solves this by asking a different question: not "does the
judge agree with the probe's deception verdict?" but **"is this model an
organism?"** Organisms (adapter-modified models) carry a detectable
distributional shift in their residual stream — independent of whether the
specific answer is honest or deceptive. The organism head learns this shift
and the router picks the right tool:

```
if organism head fires (confident organism):
    → deception head score (probe excels on organisms)
else:
    → Phoenix v8 judge score (judge excels on base models)
```
Assumption is that probe does better when an adapter is used otherwise the judge does better.

## Architecture

```
 conversations ──► extraction ──► h_L46 ──► TransformerTokenProbe (L46)
                                                       │
   ┌──────────────────────────┐                       │
   │   projection: h → 128    │                       │
   │   + positional encoding  │                       │
   │   + Transformer (2 blk)  │                       │
   │   + mean-pool            │                       │
   │          │               │                       │
   │     pooled (128)         │                       │
   │       ┌──┴──┐            │                       │
   │       ▼     ▼            │                       │
   │  deception  organism     │                       │
   │  Linear     Linear       │                       │
   │       │       │          │                       │
   │       ▼       ▼          │                       │
   │     σ(·)     σ(·)        │                       │
   │       │       │          │                       │
   └───────┼───────┼──────────┘                       │
           │       │                                  │
           ▼       ▼                                  │
    probe_score   org_confidence                      │
           │       │                                  │
           │       ▼                                  │
           │  ┌──────────────┐                        │
           │  │    router    │                        │
           │  │              │                        │
           │  │ org_conf     │                        │
           │  │  > threshold?│                        │
           │  │   │      │   │                        │
           │  │  yes    no   │                        │
           │  │   │      │   │                        │
           │  │   ▼      ▼   │                        │
           │  │ probe  judge │                        │
           │  └───┬──────┬───┘                        │
           │      │      │                            │
           └──────┘      │                            │
                         ▼                            ▼
                    final_score
```

### Components

1. **Transformer encoder** (existing, frozen): processes L46 residual stream
   tokens. Projects hidden_dim → d_model=128, adds sinusoidal positional
   encoding, runs 2 transformer blocks, mean-pools over response tokens.
   Produces one `pooled ∈ ℝ^128` vector per example.

2. **Deception head** (existing, frozen): `Linear(128 → 1)`. Trained to
   classify honest vs deceptive on Metis instruct-deception data. Output: `probe_score = σ(logit)` ∈ [0,1] — BCE-trained probability, no standardisation.

3. **Organism head** (new): per-family `Linear(128 → 1)` loaded from
   `{qwen,gemma}_organism_head.pt` with per-family StandardScaler folded in.
   Trained as LogisticRegression on current encoder's pooled features (8000 rows),
   Qwen 5-fold CV AUROC 0.93, Gemma 0.61. Per-family projection layers handle
   different hidden_dims → same ℝ^128. Output: `org_confidence = σ(logit)` ∈ [0,1]
   — BCE-trained probability, no standardisation.

4. **Phoenix v8 judge**: Qwen3.5-9B with adapter
   `Jazhyc/aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2` (Kimi K3
   full binary-soft-distilled r16). Multi-route (HP-KR base model for harm rows,
   action specialist, intent specialist). Outputs a probability ∈ [0,1] via
   `score_messages()` — no logit standardisation.

5. **Router** (new, replaces sign gate): hard switch.

### Router

```
if organism_conf > ORGANISM_THRESHOLD:
    final_score = probe_score    # organism → deception head
else:
    final_score = judge_scores   # base model → Phoenix v8 judge
```

Both branches are probabilities ∈ [0,1]:

- `probe_score` ∈ [0,1]: sigmoid of the raw deception logit from the L46
  probe head. No standardisation — BCE-trained probability.
- `judge_scores` ∈ [0,1]: probability from Phoenix v8's `score_messages()`.
- `organism_conf = σ(organism_logit)`: sigmoid of the **raw** organism head
  logit. No standardisation — the BCE loss directly calibrates the sigmoid
  output as P(organism). The threshold is a natural probability.

- `ORGANISM_THRESHOLD` (env var, default 0.7): rows with P(organism) > 0.7
  use the deception head; others use the judge.
- No blending, no interpolation. One score source per row.

### Why L46 only (not dual L40+L46)

v6 used dual probes fused as `(z_46 + z_40) / 2.0`. v7 drops to L46 only:

1. **Shared encoder is cleaner.** The organism head needs an encoder to
   produce `pooled ∈ ℝ^128`. Sharing the L46 encoder means zero extra
   extraction cost and zero extra parameters in the trunk.
2. **L46 alone is already strong.** v6-mini-long's L46 component scores Iris
   AUROC ~0.99 and the dual fusion only adds ~0.002 on Metis. The gain from
   the organism router (correctly routing mo-qwen organisms to the probe) dwarfs
   the fusion gain.
3. **Half the extraction volume.** Extracting only L46 instead of L40+L46
   cuts extraction time roughly in half — significant for the 30-min budget.

## Organism head training

**Method:** Option A — freeze the existing L46 encoder + deception head,
forward-pass all Metis qwen+gemma rows, collect pooled(128) features, train
per-family LogisticRegression with `class_weight='balanced'` and per-family
`StandardScaler`. Scaler folded into Linear weights at load time so forward
pass is a single matmul.

**Data:** Full Metis (7200 qwen + 800 gemma rows), no holdout. Labels:
`is_organism = (lora is not None)`. Nemotron excluded (no organism rows).

**Evaluation:** 5-fold stratified CV. Per-family heads compared against a
single shared head trained on qwen+gemma combined features.


## Router behaviour by dataset

### Notus (base models — Nemotron, Qwen, Gemma, no adapters)

| organism_conf | router → | score source | expected AUROC |
|---|---|---|---|
| < threshold (most rows) | judge | judge_scores | ~0.89 |
| > threshold (false positives) | probe | probe_score | ~0.56 ⚠️ |

False positives are the main risk — base-model rows where the organism head
fires and the probe takes over with its ~0.56 AUROC. The threshold should be
set so false positives are rare. If needed, `ORGANISM_THRESHOLD` can be
bias toward the judge (fewer false positives, more false
negatives on Iris — an acceptable trade since the judge scores ~0.96 on Iris).

### Iris (organisms — g-st adapters)

| organism_conf | router → | score source | expected AUROC |
|---|---|---|---|
| < threshold (most rows) | judge | judge_scores | ~0.96 |
| > threshold (few rows) | probe | probe_score | ~0.99 |

The organism head was trained on Metis mo-qwen adapters and doesn't generalise
to g-st adapters — only ~8% of Iris g-st rows clear the threshold. This is
acceptable: the judge scores ~0.96 on Iris, so the penalty is only ~0.03 AUROC
vs routing to the probe. It's a missed opportunity, not a bug.

## Results (Jul 29, 2026)

All results below are **local evaluations on Metis dev data** (21 datasets, 8216 rows,
cached L46 activations). The probe was trained on all Metis data (internal 85/15
split for early stopping), so local eval is in-distribution. On Notus base models
(OOD), the probe is expected to drop to ~0.56. The organism head was trained on
all 8000 qwen+gemma pooled features from the same probe; 5-fold CV numbers (Qwen
0.93, Gemma 0.61) are reported alongside full-data AUROC. Iris g-st results are
from Metis dev g-st splits (a proxy, not the actual Iris leaderboard dataset).

### Organism head calibration

The Qwen organism head is **well-calibrated** on Metis. The folded bias of +4.56
is offset by negative dot products on base rows, producing negative logits
(mean −1.62) and low confidences (mean 0.19). At t=0.7, **zero** Metis Qwen base
rows are misrouted to the probe.

The organism signal doesn't generalize across adapter types — the head was
trained on Metis mo-qwen adapters (a-mo, b-mo, c-mo) but Iris uses g-st
adapters. Empirically tested on cached activations:

| data | Qwen conf (mean) | % > t=0.5 | % > t=0.7 | routes to |
|---|---|---|---|---|
| Metis BASE (instructed) | 0.19 | 1.2% | **0%** | judge ✅ |
| Metis BASE (varied) | 0.15 | 2.8% | 0.5% | judge ✅ |
| Metis ORG a-mo-1 | 0.91 | 99.2% | **93.5%** | probe ✅ |
| Metis ORG b-mo | 0.92 | 99.0% | 93.0% | probe ✅ |
| **Metis dev G-ST** (Iris proxy) | **0.25** | **8.2%** | **3.2%** | **judge** ⚠️ |

The head fails to detect Metis dev g-st organisms (proxy for Iris) — they route
to the judge. This is acceptable: the Phoenix v8 judge scores ~0.96 on Iris, so
the penalty is only ~0.03 AUROC vs routing to the probe (~0.99).

Gemma is weak as expected (CV AUROC 0.61). At t=0.7 the false-positive rate on
Gemma base rows is controlled (0.8%), but true-positive rate is poor (8.5%).
Gemma g-st detection is near-zero (2.8%). Again, the judge handles these rows
adequately.

Notus Qwen base behavior is unmeasured (no cached Notus activations available)
but the organism logit for Metis base Qwen rows averages −1.62 — well below zero.
Unless Notus Qwen features shift dramatically in the organism direction, the head
should remain quiet on Notus base rows too.

### Offline organism head retraining

Trained on current L46 encoder pooled features. Per-family StandardScaler,
`class_weight='balanced'`, 5-fold stratified CV. Qwen: 7200 rows (800 base,
6400 org). Gemma: 800 rows (400 base, 400 org). Combined: 8000 rows.

| | Qwen CV (AUROC) | Gemma CV (AUROC) | Full-data (AUROC) |
|---|---|---|---|
| **Per-family** | **0.930** ± 0.007 | 0.614 ± 0.021 | Qwen 0.939, Gemma 0.671 |
| Shared | 0.809 ± 0.009 | — | 0.823 |

**Decision: per-family.** +0.12 AUROC on Qwen (90% of data). Gemma is weak
either way (only 800 rows). Per-family StandardScaler folded into Linear weights
at load time — forward pass is still a single matmul per row.

### Organism head quality (all 21 datasets)

| | Qwen (AUROC) | Gemma (AUROC) | Global (AUROC) |
|---|---|---|---|
| AUROC | 0.939 | 0.677 | 0.929 |
| FPR (base→probe) | 1-3% | 14% | 16.2% |
| TPR (org→probe) | 92-100% | 29% | 85.8% |
| Precision | — | — | 96.2% |

Known weakness: c-mo organisms detected at only 2-6% (router sends to judge).

### v7 router vs components (all 21 Metis dev datasets)

Local evaluation with Phoenix v6.3 multi-route judge (proxy — the v8 Kimi K3
was not available for local eval at the time).
8216 rows, t=0.5 unless noted.

| Method | All 21 datasets (AUROC) | Base models only (AUROC) | Organisms only (AUROC) |
|--------|-----------------|------------------|----------------|
| Judge only (v6.3 full multi-route) | 0.960 | 0.989 | 0.953 |
| Probe only (L46) | 0.992 | 0.987 | 0.993 |
| **v7 router t=0.5 (v6.3 multi-route judge)** | **0.997** | **0.992** | **0.998** |
| **v7 router t=0.7 (v6.3 multi-route judge)** | **0.997** | **0.990** | **0.998** |
| Oracle (ground-truth labels) | 0.963 | 0.835 | 0.993 |

With the full multi-route judge, the router handily beats oracle — it picks probe
(0.993) for organisms and judge (0.989) for base models, but also routes within
organism datasets: when the organism head fires, probe takes over; when confidence
is low (harder deception cases, c-mo adapters), it falls back to the strong judge.
The head is well-calibrated: on Metis, Qwen mo-qwen organism rows score ~0.99
and route to the probe, while Qwen base rows score ~0.19 and route to the judge.
t=0.5 and t=0.7 produce near-identical results because the head outputs are
already well-separated — threshold changes only affect the small Gemma subset.

### Shared vs per-family organism head (v6.3 single-route judge, t=0.5)

| | v7 per-family (AUROC) | v7 shared (AUROC) | Δ (AUROC) |
|---|---------------|-----------|----|
| All datasets | 0.950 | 0.949 | +0.001 |
| Base models | 0.850 | 0.861 | −0.012 |
| Organisms | 0.973 | 0.970 | +0.004 |
| Qwen | 0.959 | 0.952 | +0.007 |
| Gemma | 0.905 | 0.962 | −0.057 |
| Nemotron | 0.868 | 0.868 | 0.000 |

Nemotron has no organism head — no training data (all 216 rows are base model).
Applying a head trained on qwen+gemma features would produce random confidences
(different projection layers → different feature space). Judge-only is the safe
default: no risk of routing base rows to a weak OOD probe.

Aggregate is a near-tie. Shared "wins" on Gemma because it routes 45% of gemma
base rows to the probe vs per-family's 14% — on Metis training data, the probe
beats the judge. On OOD Notus where probe scores ~0.56, routing base rows to the
probe would be catastrophic. Per-family is safer: lower FPR, wins on Qwen,
identical cost.

### Caveats

All local results are on Metis dev data — in-distribution for both the probe and
organism head. On OOD Notus base models the probe is expected to drop to ~0.56.
Iris g-st results are from Metis dev g-st splits (proxy, not the actual Iris
leaderboard dataset). No Notus or Eunomia cached activations available locally.
Full multi-route judge cache:
`experiments/ensemble_gate_eval/v63_full_judge_cache.json`.
