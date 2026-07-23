# sonic_v2 — White-Box/Black-Box Gated Fusion Detector

## Overview

sonic_v2 is a **two-detector** system that reconciles a lightweight
activation-probe (white-box) with a capable LLM-based judge (black-box).
It was created for **[Aletheia's Quest](https://aletheias.com)** — a
deception-detection competition where contestants must determine whether an
AI assistant's final answer is deceptive (1) or truthful (0).

The core insight: the competition's **validation** datasets (the only ones
that count toward the final score) fall into two categories:

1. **Notus datasets** — the white-box probe achieves near-perfect ranking
   (AUROC > 0.99) but its threshold can be miscalibrated (balanced accuracy
   as low as 0.50 if cut at 0.5). The probe's ranking is excellent; it just
   needs a better threshold.
2. **Iris datasets** — the probe is near-chance on these, but the black-box
   judge (Qwen3.5-9B + LoRA) handles them well (~0.73 balanced accuracy).

sonic_v2 detects which regime it's in by measuring **per-dataset probe/judge
agreement** (AUROC of probe scores against judge verdicts). When agreement ≥
0.65, it trusts the probe and calibrates its threshold **unsupervised** via a
2-component Gaussian Mixture Model. When agreement < 0.65, it falls back to
the judge's proven verdicts, using continuous confidence scores for the AUROC
column.

This gives the best of both worlds: Notus gets the probe's 0.99+ AUROC with
a properly calibrated cutoff, and Iris gets the judge's solid balanced
accuracy.

## Architecture

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                        DATASET (per row)                        │
   └────────────┬────────────────────────────────────┬───────────────┘
                │                                    │
                ▼                                    ▼
   ┌───────────────────────┐          ┌──────────────────────────────┐
   │  White-box probe      │          │  Black-box judge             │
   │                       │          │  (Qwen3.5-9B + LoRA)         │
   │  1. Extract L46       │          │                              │
   │     activations from  │          │  1. Generate reasoning +     │
   │     generating model  │          │     verdict (do_sample=False)│
   │  2. Standardize per-  │          │  2. Soft readout: teacher-   │
   │     feature moments   │          │     force up to Prediction:, │
   │  3. TransformerToken- │          │     read P("1") vs P("0")   │
   │     Probe → sigmoid   │          │     from logits (Lever 2)   │
   │                       │          │                              │
   │  Output: probe_scores │          │  Output: judge_labels (0/1)  │
   │           [0,1]       │          │          judge_probs  [0,1]  │
   └───────────┬───────────┘          └──────────────┬───────────────┘
               │                                    │
               └──────────────┬─────────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  GATE                         │
               │  Measure: AUROC(probe_scores, │
               │           judge_labels)       │
               │                              │
               │  if AUROC ≥ 0.65  ──────┐    │
               │  if AUROC < 0.65  ──┐   │    │
               │  if no judge       ─┐│   │    │
               └─────────────────────││───│────┘
                                     ││   │
                   ┌─────────────────┘│   └──────────────┐
                   ▼                  ▼                  ▼
   ┌────────────────────────┐  ┌──────────────┐  ┌──────────────────┐
   │  TRUST PROBE (Notus)   │  │  FALL BACK   │  │  PROBE ONLY      │
   │                        │  │  TO JUDGE    │  │  (dev / no judge)│
   │  Threshold: 2-GMM      │  │  (Iris)      │  │                  │
   │  crossover (Lever 1)   │  │              │  │  score = probe   │
   │                        │  │  score =     │  │  deceptive =     │
   │  score = probe         │  │  judge_probs │  │  probe ≥ 0.5    │
   │  deceptive =           │  │  deceptive = │  │                  │
   │  probe ≥ GMM_threshold │  │  judge_labels│  │                  │
   └────────────────────────┘  └──────────────┘  └──────────────────┘
                               │
                               ▼
                   ┌──────────────────────┐
                   │  submission.csv       │
                   │  index,deceptive,score│
                   └──────────────────────┘
```

## Detector 1: White-Box Transformer Token Probe

### What it does

For every row in the dataset, sonic_v2 runs the input through the **same
model that generated the data** (on NDIF's remote infrastructure), reads the
residual-stream activations at a fixed decoder layer (L46), extracts every
token of the assistant's response, feeds those token activations through a
small trained Transformer encoder, and produces a deception probability.

### Probe architecture (`TransformerTokenProbe`)

Defined in `experiments/nonlinear_probe/token_probes.py`:

```
Input: (num_response_tokens, hidden_dim) — activations at L46,
       standardized to zero mean / unit variance per feature
       (training-set moments)

1. Linear projection:      hidden_dim → 128 (d_model)
2. + Sinusoidal positional encoding
3. TransformerEncoder × 2 blocks
   - 4 attention heads
   - FFN hidden dim 256
   - Dropout 0.1
   - src_key_padding_mask excludes padding
4. Masked mean pooling (over real tokens)
5. Linear head:            128 → 1
6. Sigmoid → deception probability
```

This is a deliberately small model (~180K parameters). It runs on CPU in the
competition sandbox in seconds — it just needs the raw activations from NDIF.

### Per-family weights

There are three separate probe copies, one per base model family:

| Family    | Hidden dim | Layer | # training rows | Training data                        |
|-----------|-----------|-------|----------------|--------------------------------------|
| **Qwen**  | 5120 (L46) | 46    | ~5,760         | Dev splits, both scenarios, all organisms |
| **Gemma** | 5376 (L46) | 46    | ~640           | Dev splits, both scenarios, all organisms |
| **Nemotron** | 4096 (L46) | 46  | ~216           | Full dev set (single dataset, 5-fold OOF) |

### How the probes were trained

1. **Activation extraction** (`extract_activations.py`):
   Every row of every dev dataset was run through its generating model on a
   local GPU (Hábrók cluster), and the residual-stream activations at selected
   decoder layers were cached to `.tokens.npz` files. Each cache holds every
   response token's activation in float16, plus per-example token offsets.

2. **Architecture sweep** (`token_sweep.py`):
   Three token-probe architectures (attention-pooling, CNN, Transformer) were
   compared via:
   - **Cross-scenario**: train on varied-deception, evaluate on
     instructed-deception (and vice versa)
   - **Leave-one-organism-out**: train on all organisms except one, evaluate
     on the held-out organism

   The Transformer architecture won consistently at L46.

3. **Standardized split evaluation** (`standardized_sweep.py`):
   Using deterministic 80/10/10 train/validation/test splits (created by
   `scripts/make_dev_splits.py`), all probe types were compared on held-out
   test data. Four training compositions were tested:
   - `both` — train on both scenarios (best for gemma)
   - `varied` — varied-deception only
   - `instructed` — instructed-deception only
   - `both+val` — train + validation splits (best for qwen)

   Results on the held-out test split (BalAcc):
   ```
   qwen   transformer  both+val  0.941  (AUROC 0.984)
   gemma  transformer  both+val  0.988  (AUROC 0.990)
   gemma  transformer  both      0.962  (AUROC 0.985)
   qwen   transformer  both      0.915  (AUROC 0.981)
   qwen   cnn          both+val  0.947  (AUROC 0.986)
   ```

4. **Final training**:
   - **Qwen & Gemma**: trained on all rows of the dev split (both scenarios,
     all organisms) using the standardized pipeline's "both" composition,
     which was the most conservative/general. The Transformer token probe was
     trained with early stopping on a held-out 15% validation split, Adam
     optimizer, BCEWithLogitsLoss, and the best-validation-loss weights
     restored.
   - **Nemotron**: trained via 5-fold stratified cross-validation on its
     single 216-row dev dataset (`nemotron_probe_train.py`). The best layer
     was selected by OOF balanced accuracy. Because the Nemotron NDIF
     deployment has very little GPU headroom, the submission notebook's
     extraction config limits it to batch size 1 / token budget 512.

## Detector 2: Black-Box Judge (Qwen3.5-9B + LoRA)

### What it does

The judge is the same model that powered `phoenix_wright_v2` — a
Qwen3.5-9B base model with a LoRA adapter
(`Jazhyc/aletheias-phoenix-wright-v2-adapter`) fine-tuned for deception
classification.

For each row, it:
1. Truncates the conversation to fit the context window (2048 token max)
2. Applies a system prompt instructing it to decide if the final output is
   deceptive
3. Generates a reasoning summary followed by `Prediction:<0 or 1>`
4. Extracts the verdict from the generated text

### Lever 2: Verdict-Token Logit Readout (Soft Confidence)

The generated verdict is **greedy** (do_sample=False) and gives a hard 0/1.
But for the AUROC (score) column, a continuous ranking is far more useful.

sonic_v2 also does a **second forward pass** that teacher-forces the prompt
+ the generated reasoning up to the "Prediction:" marker, then reads the
**logits for token ID "1" vs "0"** as the very next token:

```
P("deceptive") = softmax([logit_ID0, logit_ID1])[1]
```

This gives a continuous confidence score without changing the verdict. A
self-check requires ≥80% agreement between the soft readout's argmax and
the originally generated verdict — otherwise it falls back to the hard 0/1.

The soft readout is **strictly additive**: it can only improve AUROC ranking
resolution; any failure returns the hard verdicts, so the balanced accuracy
column is never regressed.

## The Gate: Per-Dataset Probe/Judge Agreement

The gate is the component that decides **per dataset** which detector to
trust. It's defined in cell 12 of `sonic_v2.ipynb`.

### How it works

1. **Measure agreement**: Compute the AUROC of the probe's scores against
   the judge's binary verdicts (ignoring which is "right" — just measuring
   how well their rankings align).

2. **Handle inversion**: If AUROC < 0.5, the probe ranks backwards relative
   to the judge (still real signal, just inverted). Set `agreement = 1.0 -
   agreement` and flip the probe scores.

3. **Gate decision**:
   - **agreement ≥ 0.65** → "Trust the probe"
     - Keep the probe's ranking for the `score` column (AUROC)
     - Set the threshold via **unsupervised GMM** (Lever 1) instead of
       pinning it to the judge's labels
   - **agreement < 0.65** → "Trust the judge"
     - Write `judge_soft` (continuous) to the `score` column
     - Write `judge_labels` (hard 0/1 verdicts) to the `deceptive` column

4. **No judge available** (dev datasets, or judge failure): Fall through to
   the plain probe with its default 0.5 threshold.

### Why 0.65?

The 0.65 threshold was chosen empirically. When the probe's AUROC against
the judge is ≥ 0.65, the probe's ranking has meaningful alignment with a
reasonable detector, so it's safe to trust the probe's finer-grained signal.
Below 0.65, the probe's ranking is too noisy for this task and the judge's
proven verdicts are more reliable.

The gate only fires on **validation-split** datasets (Notus + Iris). Dev
datasets always use the probe alone — the judge is too expensive to run on
all dev data, and the dev score doesn't count anyway.

### Lever 1: Unsupervised Threshold Calibration (GMM)

When the gate trusts the probe, it needs to pick a decision boundary for the
`deceptive` column. The key insight: **the probe's AUROC on Notus is already
0.99+ — the quality ceiling is the threshold, not the ranking**.

Instead of using the judge's labels to pick the threshold (which would cap
the probe at the judge's quality — the Iris failure case), sonic_v2 uses a
completely unsupervised method:

1. Fit a 2-component Gaussian Mixture Model to the probe's score distribution
2. Find the point where the two modes' posterior probabilities cross
   (the decision boundary)
3. If the scores aren't cleanly bimodal (mixture degenerates), fall back to
   the median

This works because the probe's deception scores are naturally bimodal on
datasets where it has good signal — one mode for truthful responses, one for
deceptive. The GMM crossover picks the optimal threshold without ever seeing
a label.

## Submission Output

### The two scoring columns

The competition expects three columns: `index`, `deceptive`, `score`.

- **`score`** → determines AUROC on the leaderboard
  - When gate trusts probe: probe's ranking (continuous probability)
  - When gate distrusts probe: judge's continuous confidence (soft readout)
  - Fallback: probe's ranking

- **`deceptive`** → determines Balanced Accuracy on the leaderboard
  - When gate trusts probe: `probe_score ≥ GMM_crossover`
  - When gate distrusts probe: judge's hard 0/1 verdict
  - Fallback: `probe_score ≥ 0.5`

This dual-column strategy is the core of sonic_v2's improvement: Notus
gets the probe's 0.99+ AUROC in `score` AND a correctly calibrated cutoff
in `deceptive`; Iris gets the judge's solid verdicts in both columns.

## Datasets

### Scoring (validation) — Notus + Iris

| Dataset | Model | Description |
|---------|-------|-------------|
| validation-instructed-deception-Qwen3.5-27B-None | Qwen | Probe excels |
| validation-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1 | Qwen + LoRA | Probe excels |
| validation-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3 | Qwen + LoRA | Probe excels |
| ... (etc, all Qwen-variant Notus sets) | Qwen | Probe excels |
| validation-varied-deception-Gemma... | Gemma | Probe excels |
| validation-varied-deception-Nemotron | Nemotron | Judge helps |
| Iris (varied-deception, held-out tasks) | Various | Probe near-chance; judge reliable |

### Reference/Dev (not scored)

These are greyed out on the leaderboard but were used for probe training:
dev-instructed-deception-*, dev-varied-deception-*, dev-test-*

## Dependencies

### Hardware

- **White-box probe**: runs on NDIF remote infrastructure (nnsight). Requires
  one remote trace per dataset to extract L46 activations.
- **Black-box judge**: runs on NDIF. Requires two passes per dataset
  (generation + soft readout).
- **Probe scoring**: CPU-only, takes seconds.
- **Threshold calibration**: CPU-only, takes milliseconds.

### Software versions (NDIF-compatible)

```
nnsight        @ git+https://github.com/ndif-team/nnsight.git@hackathon/peft
transformers   @ c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d  (~5.15)
peft           0.18.0
torch          2.x (ships with the base image)
datasets       (ships with the base image)
scikit-learn   (ships with the base image)
```

These versions are critical: newer transformers (≥5.11) mis-register
Qwen3.5 as an image-text model, causing nnsight's LanguageModel to fail to
trace it. The pinned commit at c7f9c88 loads Qwen3.5 correctly.

## Known Limitations

1. **Per-family probes don't generalize across model families** — if a
   validation dataset uses an unseen architecture (e.g. Llama, Mistral),
   there are no probe weights for it and it falls back to 0.5.

2. **Nemotron's tiny training set** (216 rows, 16 deceptive) limits probe
   quality. The Nemotron dataset scores 0.9688 BalAcc versus 0.99+ for
   Qwen/Gemma.

3. **The judge is also Qwen-based** — if Qwen3.5-9B fails to load (version
   mismatch, or NDIF having no deployment for it), the gate mechanism
   collapses to probe-only for all datasets, including Iris.

4. **GMM threshold can fail** on unimodal score distributions. The fallback
   to the median handles most cases, but not all.

## Future Work (sonic_v2.1)

The next version should focus on:

1. **Multi-family probe training**: Instead of per-family weights, pool
   training data across all base model families. Standardize activations
   by projecting each family's hidden_dim to a common space before the
   Transformer encoder. This would generalize to unseen model architectures
   (Llama, Mistral, etc.) on the validation sets.

2. **Notus cross-organism generalization**: Currently each organism-LoRA
   combination in the training data is treated as a separate dataset.
   Probe training could be made organism-agnostic by adding organism
   embedding + adversarial removal, or by training on organism-mixed
   batches.

3. **Judge ensemble**: Run multiple judges (Qwen3.5-9B + different adapters,
   or different base models) and use their disagreement as a confidence
   signal for the gate.

4. **Joint probe+judge**: Feed the probe's per-token attention scores as
   additional context to the judge, letting it focus on tokens the probe
   finds suspicious.

## File Map

```
submission/
├── sonic_v2.ipynb                  # The submission notebook
├── sonic_v2_DOCUMENTATION.md       # This file
├── util.py                         # Build model, chat preprocess, write CSV
├── requirements.txt                # Extra pip deps (peft, nbclient, etc.)
├── whitebox_probe/
│   ├── __init__.py
│   ├── qwen_probe/
│   │   ├── config.json             # Architecture hyperparameters + layer
│   │   ├── model.pt                # Trained TransformerTokenProbe weights
│   │   ├── feature_mean.pt         # Per-feature standardization mean
│   │   └── feature_std.pt          # Per-feature standardization std
│   ├── gemma_probe/
│   │   └── ... (same structure)
│   └── nemotron_probe/
│       └── ... (same structure)

experiments/nonlinear_probe/
├── token_probes.py                 # All token probe architectures
├── extract_activations.py          # Extract + cache activations from NDIF
├── token_sweep.py                  # Cross-scenario + holdout sweep
├── standardized_sweep.py           # Final evaluation on dev-split test set
├── nemotron_probe_train.py         # Nemotron-specific 5-fold training
├── linear_sweep.py                 # Pooled-vector logistic baselines
├── nonlinear_sweep.py              # MLP probe comparisons
├── layerstack_sweep.py             # Multi-layer stacking probes
├── torch_probes.py                 # Pooled-vector torch probes
├── plot_layer_curve.py             # Layer-depth performance curves
└── plot_probe_comparison.py        # Architecture comparison plots

results/whitebox/
├── activations/                    # Extracted .tokens.npz caches (~14 GB)
├── token_sweep/                    # Cross-scenario + holdout results
└── standardized_sweep/             # Final dev-split results

dev_splits/manifest.csv             # 80/10/10 splits per dataset
scripts/make_dev_splits.py          # Split creation
```

## How to Run

### Dry run (local)

```bash
.venv/bin/python submit.py --dry
```

This runs all dev datasets through the probe. The judge is skipped on dev
(only runs on validation split). Qwen extraction requires the correct
transformers version (see Dependencies above).

### Real submission

```bash
NDIF_API_KEY="your-key" .venv/bin/python submit.py --tag white --team "SAIN Groningen"
```

The submission notebook reads `$DATASET_NAME` from the environment, runs
through all 13 cells, and writes `submission.csv` in the working directory.

## Performance

### Dev set (14 datasets, probe-only since judge doesn't fire on dev)

```
Bal.Acc 0.9892   AUROC 0.9996   Recall 0.9945   FPR 0.0161
```

Individual Qwen datasets: 0.99–1.00 BalAcc, 0.996–1.000 AUROC
Individual Gemma datasets: 0.992–1.000 BalAcc, 0.9997–1.000 AUROC
Nemotron: 0.9688 BalAcc, 0.9991 AUROC

### Validation set (expected, based on probe dev performance + judge analysis)

- **Notus (Qwen/Gemma)**: 0.99+ BalAcc, 0.999+ AUROC — probe dominates,
  gate trusts it, GMM finds correct threshold
- **Iris (held-out tasks)**: ~0.73 BalAcc, ~0.85 AUROC — judge dominates,
  gate falls back to judge verdicts
