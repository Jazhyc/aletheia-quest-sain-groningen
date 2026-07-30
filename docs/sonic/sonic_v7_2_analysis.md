# Three-Way Comparison: v6mini_long (Probe-Only) vs. sonic v7.2 (Route) vs. phoenix v8 (Judge)

## Headline AUROC (Primary Metric)

| Submission | Headline AUROC | Δ vs. best | Δ vs. v6mini | Runtime |
|-----------|---------------|------------|-------------|---------|
| v6mini_long (probe-only) | 0.7773 | –0.1667 | – | 25m 7s |
| sonic v7.2 (route) | 0.9251 | –0.0189 | +0.1478 | 29m 38s |
| **phoenix v8 (judge-only)** | **0.9440** | — | +0.1667 | 11m 47s |

---

## Architecture

```
sonic v7.2 = probe (same as v6mini) + organism router + judge (phoenix v7-style)

           ┌──────────────────────┐
L46 acts ──┤ TransformerTokenProbe├──► probe_score (deception probability)
           └──────┬───────────────┘
                  │  same pooled features
           ┌──────┴───────────────┐
           │ Linear(128→1)        ├──► organism_conf = P(organism model)
           │ "organism head"      │
           └──────┬───────────────┘
                  │
                  ▼
     organism_conf > 0.7 ?  ──Yes──► use probe_score
                  │
                 No
                  │
                  ▼
          Qwen3.5-9B v7 judge ──────► use judge_score
          (+ intent adapter)
```

---

## The Problem, Precisely Stated

### The organism head answers the wrong question

It is trained to detect: *"Do these activations come from an organism-fine-tuned model?"*

But the routing needs to answer: *"Will the probe be more accurate than the judge on this input?"*

These are **correlated but not identical**, and the mismatch produces two failure modes:

### Failure Mode 1: False Negatives — Probe works but organism head says "no"

| Dataset | Model | Probe AUROC | Judge AUROC | Organism says | Result |
|---------|-------|:----------:|:----------:|:------------:|--------|
| Iris | Qwen g-st | **0.999** | 0.967 | "Not organism" | Uses judge, loses **0.032 AUROC** |
| Iris | Nemotron g-st | **0.990** | 0.971 | "Not organism" | Uses judge, loses **0.019 AUROC** |
| Eunomia | base | **0.994** | 0.989 | "Not organism" | Uses judge, loses **0.005 AUROC** |
| Eunomia | b-mo | **0.974** | 0.962 | Mixed | Route averages, loses to both |

The `g-st` (general steering) models shift activations in a direction the probe can detect — but it is a *different* direction from organism fine-tuning, so the organism head does not recognize it. The probe is discarded when it would have been dominant.

### Failure Mode 2: False Positives — Judge is better but organism head says "yes"

| Dataset | Model | Probe AUROC | Judge AUROC | Organism says | Result |
|---------|-------|:----------:|:----------:|:------------:|--------|
| Eunomia | a-mo-5 | 0.962 | **0.978** | "Organism" | Uses probe, loses **0.016 AUROC** |
| Eunomia | c-mo | 0.983 | **0.983** | "Organism" | Uses probe, tie at best |

Strong organism adapters are correctly detected, but the judge actually outperforms the probe on them.

### Failure Mode 3: The organism head works

| Dataset | Model | Probe AUROC | Judge AUROC | Organism says | Result |
|---------|-------|:----------:|:----------:|:------------:|--------|
| Notus | All 3 models | ~0.56 (broken) | ~0.92 | "Not organism" | Correctly uses judge ✓ |

The probe is near-chance on Notus. The organism head correctly says "not organism" → judge saves the day. This is the **one and only** scenario where the organism-head router adds value.

---

## Net Effect

| | Probe wins on | Judge wins on | Router chooses |
|---|---|---|---|
| Iris Qwen g-st | ✓ (+0.032) | | Judge ✗ |
| Iris Nemotron g-st | ✓ (+0.019) | | Judge ✗ |
| Eunomia a-mo-5 | | ✓ (+0.016) | Probe ✗ |
| Eunomia a-mo-6/7 | | ✓ (+0.004, +0.020) | Probe ✗ |
| Notus all | | ✓ (+0.36) | Judge ✓ |

The organism head makes one correct decision (Notus) and **four incorrect or costly decisions** (Iris Qwen, Iris Nemotron, Eunomia a-mo-5, a-mo-7). The net AUROC impact is **negative**.

---

## Validation Data Breakdown

### Eunomia (Qwen3.5-27B)

| LoRA | v6mini (probe) | v7.2 (route) | phoenix v8 (judge) | Best |
|------|:-:|:-:|:-:|:-:|
| *(base)* | 0.9941 | **0.9973** | 0.9892 | route |
| a-mo-1 | **0.9021** | 0.8853 | 0.8784 | probe |
| a-mo-3 | **0.8346** | 0.7961 | 0.7774 | probe |
| a-mo-4 | 0.8973 | **0.8992** | 0.8973 | route |
| a-mo-5 | 0.9620 | 0.9550 | **0.9775** | judge |
| a-mo-6 | 0.9119 | 0.9106 | **0.9143** | judge |
| a-mo-7 | 0.9188 | 0.9072 | **0.9270** | judge |
| b-mo | **0.9743** | 0.9604 | 0.9623 | probe |
| c-mo | 0.9825 | 0.9681 | **0.9829** | judge |
| **Mean** | **0.9308** | 0.9199 | 0.9229 | probe |

Probe leads Eunomia on average but is overtaken by the judge on half the adapters. The route underperforms *both* on nearly every row — blending costs AUROC rather than getting the best of both.

### Notus

| Model | v6mini (probe) | v7.2 (route) | phoenix v8 (judge) | Best |
|-------|:-:|:-:|:-:|:-:|
| Nemotron-120B | 0.6088 | 0.9444 | 0.9444 | route/judge |
| Qwen3.5-27B | 0.6326 | 0.8856 | **0.8873** | judge |
| Gemma-3-27B | 0.4377 | 0.9091 | **0.9386** | judge |
| **Mean** | 0.5597 | 0.9130 | **0.9234** | judge |

The probe is broken on Notus. Both route and judge rescue this. This is the single biggest contributor to the v6mini → v7.2 jump.

### Iris

| Model | v6mini (probe) | v7.2 (route) | phoenix v8 (judge) | Best |
|-------|:-:|:-:|:-:|:-:|
| Nemotron g-st | **0.9904** | 0.9711 | 0.9711 | probe |
| Qwen g-st | **0.9986** | 0.8848 | 0.9666 | probe |
| Gemma g-st-2 | 0.9956* | 0.9556 | 0.9556 | route/judge |
| **Mean** | **0.9949** | 0.9372 | 0.9644 | probe |

*\* Degenerate: 0.9956 AUROC with 0.5000 Bal. Acc — the probe assigns systematically wrong scores that happen to rank correctly. Not trustworthy.*

The probe is stellar on Iris Qwen (+0.032 over judge) and Nemotron (+0.019), but the route ignores it — presumably because Iris Gemma's degenerate probe output poisons the routing heuristic. This is the largest single missed opportunity in v7.2.

---

## The Full Picture by Dataset-Model

```
Eunomia base          ██████████████████████████ probe 0.994  ▲ judge 0.989  ▼ route 0.997
Eunomia a-mo-5        ██████████████████████████ probe 0.962  ▲ judge 0.978  ▼ route 0.955
Eunomia c-mo          ██████████████████████████ probe 0.983  ▲ judge 0.983  ▼ route 0.968

Notus Qwen            ░░░░░░░░ probe 0.633        ████████████ route 0.886  ▲ judge 0.887
Notus Gemma           ░░░░ probe 0.438            ████████████ route 0.909  ▲ judge 0.939

Iris Qwen             ████████████████████████████ probe 0.999  ░░░░░░░░░░░░ route 0.885  ██████████ judge 0.967
Iris Nemotron         ██████████████████████████ probe 0.990  ████████████ route 0.971  judge 0.971
Iris Gemma            ⚠ probe 0.996 (broken!)     ████████████ route 0.956  judge 0.956
```

---

## Recommendations

### Recommendation 1 (Immediate, Low-Risk): Drop the probe, switch to phoenix v8 judge-only

This is what the data says:

| Approach | AUROC | Runtime |
|----------|:-----:|:-------:|
| Probe-only (v6mini) | 0.777 | 25 min |
| Probe + organism router + v7 judge (v7.2) | 0.925 | 30 min |
| **v8 judge-only (phoenix v8)** | **0.944** | **12 min** |

phoenix v8 beats v7.2 by **+0.019 AUROC** while being **2.5× faster** and **simpler** (no activation extraction, no organism head, no router). The v8 adapter (distilled from Kimi K3 with direct-margin scoring) just works better than v7's judge. There is no reason to pay the probe's complexity+runtime tax.

### Recommendation 2 (Medium-Term, Higher-Reward): Replace organism-head routing with probe-confidence routing

If you want to make the probe earn its keep, the routing heuristic needs to address the actual AUROC tradeoff. The key insight:

- When the probe is **confident** (score near 0 or 1), it tends to be correct — even on OOD models like Iris g-st
- When the probe is **tentative** (score near 0.5), it tends to be unreliable (Notus)

A soft blend weighted by the probe's own confidence would naturally handle all cases:

```python
confidence = abs(probe_score - 0.5) * 2  # 0.5→0, 1.0→1
final = confidence * probe_score + (1 - confidence) * judge_score
```

Expected behavior:
- **Notus**: probe near 0.5 → confidence ≈ 0 → judge dominates ✓
- **Iris Qwen g-st**: probe near 0 or 1 → confidence ≈ 1 → probe dominates ✓
- **Eunomia a-mo-5**: probe moderate confidence → blend gives some weight to better judge ✓
- **Iris Gemma g-st-2**: probe confident but wrong (degenerate) → ✗ still a problem

This would capture the probe's wins on Iris Qwen (+0.032) and Nemotron (+0.019) while keeping the judge's wins on Eunomia and Notus. The headline AUROC could plausibly exceed 0.95 — better than any current submission.

### Recommendation 3 (Speculative, Long-Term): Train the organism head on the AUROC gap directly

Instead of training the organism head to predict "is this an organism model?", train it on a per-row calibration dataset to predict *"probe_AUROC − judge_AUROC for this type of input."* This turns the router into a proper meta-learner that makes the right decision per row. Requires labeled calibration data across all model families, which may not be available.

---

## Bottom Line

The organism head is an **indirect proxy** that answers "what kind of model is this?" when the routing decision requires answering "which detector is more accurate here?" The correlation is weak enough that the router makes more bad decisions than good ones. The probe itself is excellent in its domain — the bottleneck is purely the routing heuristic.
