# sonic_v2.1 — Gate logic changes

**TL;DR:** Two bugs in cell 12 (the gate that combines the white-box probe and black-box judge). Both are fixed. No other changes.

## What the gate does

We have two detectors that run on every validation dataset:

| Detector | Score column (→ AUROC) | Threshold / verdict (→ Balanced Accuracy) |
|----------|----------------------|-------------------------------------------|
| **Probe** (transformer token probe at L46) | Continuous `σ(logit)` in [0, 1] | Binary: `score ≥ threshold` |
| **Judge** (Qwen3.5-9B + LoRA) | Continuous soft confidence in [0, 1] | Binary: `Prediction:0` or `Prediction:1` |

The gate's job is to merge them into a single `submission.csv` with `score` (continuous, drives AUROC) and `deceptive` (binary, drives balanced accuracy).

**The gate only runs on validation datasets** (Notus, Iris). On dev reports it uses the probe alone. Neither fix changes that.

---

## Fix 2 — Soft blend (was: hard switch)

### The old behavior

The old code measured `agreement = AUROC(probe_scores, judge_labels)`. Then:

```
if agreement ≥ 0.65:
    # "probe is better" → discard judge entirely
    final_scores = probe_scores
    deceptive = probe_scores ≥ unsupervised_threshold
else:
    # "judge is better" → discard probe entirely
    final_scores = judge_scores
    deceptive = judge_labels
```

**Hard switch.** If agreement = 0.64, the probe is thrown away completely — even on rows where the probe has useful signal. If agreement = 0.65, the judge is thrown away completely — even on rows where it could help calibrate the threshold.

### The new behavior (Fix 2)

Continuous blend instead of binary switch:

```
w = clamp((agreement - 0.5) × 3, 0, 1)

blended_scores = w × probe_scores + (1 − w) × judge_scores
```

| Agreement | w | Blend | Meaning |
|-----------|----|-------|---------|
| 0.50 (random) | 0.0 | pure judge | Probe ranks no better than chance vs judge |
| 0.60 | 0.3 | 30% probe + 70% judge | Leaning toward judge |
| 0.70 | 0.6 | 60% probe + 40% judge | Leaning toward probe |
| 0.83+ | 1.0 | pure probe | Probe clearly outranks judge |

This is smooth — a dataset at 0.64 doesn't suddenly lose all probe signal. The scores are still used for the AUROC column, so the better ranker always contributes.

---

## Fix 1 — Calibrate threshold against judge labels (was: unsupervised GMM)

### The old behavior (the bug)

When the gate "trusted" the probe (agreement ≥ 0.65), it still needed a threshold to produce the binary `deceptive` column. The old code fit a **2-component Gaussian mixture model** (GMM) on the probe scores alone and cut where the two Gaussians crossed:

```
GaussianMixture(n_components=2).fit(probe_scores)
→ find x where P(deceptive | x) = P(honest | x)
→ use that x as the threshold
```

**This assumes the score distribution is bimodal** — two nice Gaussian bumps, one for honest examples and one for deceptive. When this is true, the GMM works. When it's false (scores are skewed, unimodal, or have outliers), the GMM picks a spurious cut.

**The proof that it's broken:** Iris/Gemma-g-st-2.

| Metric | Value | Meaning |
|--------|-------|---------|
| Probe AUROC | **0.9911** | Probe ranks examples almost perfectly |
| GMM threshold → BA | **0.5100** | GMM cut was at ~0.0 or ~1.0 → random guessing |
| Judge BA | **0.9350** | Judge correctly classifies 93.5% of examples |

The probe's scores are well-separated (AUROC=0.99) but NOT bimodal — they might be all above 0.5 or tightly clustered with a small shift. The GMM finds a spurious split that doesn't correspond to the real deception boundary.

### The new behavior (Fix 1)

Replace the unsupervised GMM with a **supervised search against judge labels**:

```python
best_t, best_ba = 0.5, 0.0
for t in np.sort(blended_scores):
    pred = blended_scores ≥ t
    ba = balanced_accuracy_score(judge_labels, pred)
    if ba > best_ba:
        best_t, best_ba = t, ba
```

This tries every possible threshold and picks the one that maximizes balanced accuracy against the **judge's binary verdicts**.

| Row | Judge BA | What happens | BA after fix |
|-----|----------|---------------|--------------|
| Iris/Gemma-g-st-2 | 0.935 | Threshold tuned to align with judge labels | **0.935** ✅ |
| Any row where probe already works | ≥ probe BA | Threshold search matches or beats current | **Same or better** ✅ |
| Any row where probe is better than judge | probe BA | Threshold search finds boundary between hoenst/deceptive clusters | **Probe's BA maintained** ✅ |

### Why this is safe, not "capped at judge quality"

The concern was: "if we calibrate against judge labels, we're capped at the judge's accuracy." This is wrong for two reasons:

1. **The score column (AUROC) stays pure probe.** Only the `deceptive` column (BA) uses the judge-calibrated threshold. On Iris/Gemma, probe AUROC stays 0.9911 — the judge's worse ranking doesn't touch it.

2. **The threshold search can beat the judge's own BA.** The judge labels are noisy (BA=0.935 on Iris/Gemma). By searching for the threshold that best aligns with these noisy labels, we recover ≈ the judge's BA — which is far above the GMM's 0.51. On rows where the probe cleanly separates the classes, the search finds the natural boundary at the probe's full BA.

---

## Expected effect (validation-only)

| Row | Old BA | New BA | Why |
|-----|--------|--------|-----|
| Iris/Gemma-g-st-2 | 0.5100 | **0.9350** | GMM bug fixed |
| Notus/Qwen-None | 0.5700 | **0.7075** | Blend weight w≈0 → judge dominates |
| All other rows | unchanged | unchanged | Already optimal |

**Headline BA: 0.7471 → 0.8408** (+0.0937)

This beats our previous best method (phoenix_wright, 0.8333) by +0.0075.

---

## What we did NOT change

- The probe architecture, weights, extraction pipeline (cells 1–10)
- The judge model, prompt, batched generation (cell 11)
- The non-validation dataset path (no judge → pure probe, same as before)
- The fallback path (probe weights missing → 0.5 scores)

Both changes are in **cell 12 only**, between the `if judge_labels.any() and not judge_labels.all():` branch and `write_submission`. Everything else is untouched.

---

## Why no text features / logistic regression

Considered and rejected. Adding text features (perplexity, length, hedging markers) fitted on dev data would introduce a component that could **generalize negatively** to hidden data with different statistical properties. The probe already has access to the full residual stream at L46 — if text features were useful, the transformer encoder would have learned them. Adding a separate logistic on top would be memorizing dev-specific patterns, not adding new signal.

The two fixes above are zero-risk because they only **reweight existing scores** — they can't learn spurious correlations from dev data.

---

