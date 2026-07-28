# Probe confidence vs prompt length

The probe mean-pools over response tokens. Longer responses → more averaging →
probe confidence (|z|) shrinks, but accuracy (sign correctness) is unchanged.
The gate `sigmoid(judge_z + cap × probe_z)` then automatically falls back to the
judge on longer prompts. On Notus — where the judge does okay and the probe
doesn't transfer — this is **protective**.

## Experiment

**Data.** v4.3 probe logits cache: 21 dev datasets, 8,216 examples. Probe is
in-sample on every dev row — AUROC is at ceiling; the confidence correlations
are structural and should hold OOD.

**Length.** Full conversation token count (chat template, input + output) and
response-only token count (final assistant turn). Computed by loading each HF
dataset and tokenizing with the per-family tokenizer.

**Probe score.** L46 and L40 logits standardised with frozen v5b per-family
moments, fused: `z = (z_L46 + z_L40) / 2`. Confidence = `|z|` (= standard
deviations from neutral; `z` has mean ~0, sd ~1 in-distribution). Correctness =
`sign(z) == label`.

**Script:** `experiments/ensemble_gate_eval/correlate_length_probe_perf.py`

## Results

**AUROC/BA vs length: no meaningful correlation.** Per-example correctness ρ =
+0.04 (statistically "significant" but negligible effect size — <0.5%).
Per-dataset avg length vs AUROC ρ = +0.24 (p = 0.30, not significant; AUROC at
ceiling on almost every dataset). Per-length-bin AUROC is near 1.0 in every bin
from short to long. The probe ranks equally well regardless of prompt length.

What *does* correlate is confidence:

### Per-example correlation (n = 8,216)

| Metric | Full prompt ρ | Response ρ |
| --- | --- | --- |
| **Probe confidence** \|z\| | **−0.25** (p < 0.0001) | **−0.23** (p < 0.0001) |
| Probe correctness | +0.04 (p = 0.002) | +0.01 (p = 0.19) |

Confidence drops on longer prompts; correctness doesn't. The probe stays equally
*right* but speaks more *softly*.

### Per-family (full length ↔ \|z\|)

| Family | N | ρ | avg \|z\| (short) | avg \|z\| (long) |
| --- | --- | --- | --- | --- |
| **Qwen** | 7,200 | **−0.33** | 1.05 (instruct, ~100 tok) | 0.89 (varied, ~170 tok) |
| Gemma | 800 | −0.11 | 0.97 (instruct) | — |
| Nemotron | 216 | +0.01 | 0.55 | — |

Qwen shows the strongest effect: ~15% confidence drop across a 1.7× length increase.

### Per-length-bin AUROC (pooled, response tokens)

| Bin (tokens) | N | AUROC fused | BA fused |
| --- | --- | --- | --- |
| 2–44 | 5,301 | 0.9986 | 0.9892 |
| 44–87 | 1,268 | 0.9982 | 0.9911 |
| 87–129 | 845 | 0.9988 | 0.9952 |
| 129–171 | 504 | 1.0000 | 0.9982 |
| 171–214 | 197 | 1.0000 | 1.0000 |
| 214–256 | 62 | 0.9978 | 0.9660 |
| 256+ | 39 | 1.0000 | 1.0000 |

Bin midpoint ↔ AUROC: ρ = +0.57 (p = 0.11) — no significant trend. AUROC ~1.0 in
every bin. Length doesn't touch *ranking* ability; it only shrinks z-score
*magnitude*.

## Why

The TransformerTokenProbe mean-pools over response tokens:

```
probe_logit = mean_{token ∈ response} token_probe(token_embedding)
```

If each token's score is `μ + ε_i` with noise variance σ², the mean over N
tokens has variance σ²/N. The standardised z-score has magnitude ∝ 1/√N — it
shrinks toward zero on longer responses. The sign is preserved because the
expectation stays on the correct side of zero.

## Implications for Notus

This is **protective**, not harmful:

1. **The judge does okay on Notus** (~0.8458 AUROC Qwen, ~0.97 Nemotron). The
   probe degrades OOD (−0.10 to −0.15), so adding probe weight where it's wrong
   hurts.

2. **Longer prompts naturally throttle the probe.** |z| shrinks → `cap ×
   probe_z` contributes less → the gate falls back to the judge on exactly the
   rows where the probe is weakest.

3. **The sign test provides a second layer.** Even if |z| survives, the gate
   withholds MAX_CAP on disagreement rows — the probe is clamped to BASE_CAP
   regardless of confidence.

4. **Cannot produce spurious confidence.** v4.1 gated on `sigmoid(|probe_z|)`
   and lost −0.0380 on Notus Nemotron because probe confidence was inflated OOD.
   The length penalty only *shrinks* |z|, never inflates it.

**TL;DR:** Longer prompts → |z| shrinks → gate falls back to judge → method
degrades gracefully. Not a cause of Notus transfer failure — a self-regulating
mechanism that *limits* the damage.

## A possible explanation for Notus

Every dev sample fits within ~500 tokens (3,000 chars).  Notus prompts are
reportedly much larger.  If so:

- **The probe sees activations it was never trained on.**  On Notus the model
  attends over thousands of tokens — the hidden states at layer 40/46 are
  qualitatively different from the short-context activations in the training
  distribution.  This is a genuine distribution shift, not just the
  confidence-shrinking effect of token pooling.

- **The judge is truncated; the probe is not.**  The Phoenix formatter (v5.0)
  keeps only the last 3,000 chars.  On dev everything fit — on Notus the judge
  and probe would see different information.

The within-distribution ρ = −0.25 is benign.  If this theory holds, the Notus
drop (−0.10 to −0.15 AUROC) is a distribution shift in the activations
themselves, not generic OOD degradation.

## Notes

- Probe is in-sample on dev rows → AUROC at ceiling. Per-example confidence
  correlations are structural and should hold OOD.
- Uses v4.3 cache; v5b has identical probes and standardisation constants.
- Only dev datasets cached. Notus data needs a separate extraction pass.
