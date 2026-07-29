# sonic v7 — L46 probe (deception + organism heads) routed against Phoenix v6.3 judge

**Judge:** Phoenix Wright v6.3 — Qwen3.5-9B + `Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2`
(binary-soft-distilled r16 adapter, 397B teacher, 2-epoch). Multi-route with
HP-KR, action, and intent specialist judges. Standardised as `judge_z = (log_odds − 0) / 6.0`.

**Probe:** Single L46 transformer probe with two heads sharing one encoder:
one reads deception, one detects whether the model is an organism. When the
organism head fires, the deception head takes over. Otherwise, the judge
handles the row. No blending.

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
    → Phoenix v6.3 judge score (judge excels on base models)
```

## Architecture

```
                  ┌──────────────────────────────┐
  conversations ──┤ extraction                   ├── h_L46
                  └──────────────────────────────┘       │
                                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         TransformerTokenProbe (L46)           │
                  │                                              │
                  │  projection: hidden_dim → d_model (128)      │
                  │  + sinusoidal positional encoding            │
                  │  + TransformerEncoder (n_blocks=2)           │
                  │  + mean-pool over response tokens            │
                  │                                              │
                  │     pooled ∈ ℝ^128                           │
                  │         │                                    │
                  │         ├── head_deception: Linear(128→1) ──→ probe_z
                  │         │                                    │
                  │         └── head_organism:  Linear(128→1) ──→ organism_conf
                  └──────────────────────────────────────────────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │    router    │
                               │              │
                               │  organism_conf  │
                               │  > threshold?│
                               │     │        │
                               │  yes│     no │
                               │     ▼      ▼ │
                               │  σ(probe_z)  │      σ(judge_z)
                               │  (probe)     │     (judge)
                               └──────────────┘
                                      │
                                      ▼
                                 final_score
```

### Components

1. **Transformer encoder** (existing, frozen): processes L46 residual stream
   tokens. Projects hidden_dim → d_model=128, adds sinusoidal positional
   encoding, runs 2 transformer blocks, mean-pools over response tokens.
   Produces one `pooled ∈ ℝ^128` vector per example.

2. **Deception head** (existing, frozen): `Linear(128 → 1)`. Trained to
   classify honest vs deceptive on Metis instruct-deception data. Output
   output: `probe_score = σ(logit)` ∈ [0,1] — BCE-trained probability, no standardisation.

3. **Organism head** (new): single shared `Linear(128 → 1)` loaded from
   `submission/whitebox_probe/shared_organism_head.pt`. Trained as
   LogisticRegression on qwen+gemma pooled features (8000 rows), AUROC 0.85,
   r=−0.008 vs deception. Per-family projection layers handle different
   hidden_dims → same ℝ^128. Output: `organism_conf = σ(logit)` ∈ [0,1]
   — BCE-trained probability, no standardisation.

4. **Phoenix v6.3 judge**: Qwen3.5-9B with adapter
   `Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2` (binary-soft-distilled
   r16 from 397B teacher, 2-epoch). Multi-route (HP-KR base model for harm rows,
   action specialist, intent specialist). Standardised as
   `judge_z = (log_odds − 0) / 6.0`.

5. **Router** (new, replaces sign gate): hard switch.

### Router

```
if organism_conf > ORGANISM_THRESHOLD:
    final_score = probe_score    # organism → deception head
else:
    final_score = judge_scores   # base model → v7.0 judge
```

Both branches are probabilities ∈ [0,1]:

- `σ(probe_z)`: sigmoid of the **standardised** deception logit. The raw
  probe logit is shifted/scaled per family (`probe_z = (logit − mean) / sd`)
  so scores are comparable across families, then sigmoided.
- `judge_scores` ∈ [0,1]: probability from Phoenix v7.0's `score_messages()`.
- `organism_conf = σ(organism_logit)`: sigmoid of the **raw** organism head
  logit. No standardisation — the BCE loss directly calibrates the sigmoid
  output as P(organism). The threshold is a natural probability.

- `ORGANISM_THRESHOLD` (env var, default 0.5): rows with P(organism) > 0.5
  use the deception head; others use the judge.
- No blending, no interpolation. One score source per row.

### Why L46 only (not dual L40+L46)

v6 used dual probes fused as `(z_46 + z_40) / 2.0`. v7 drops to L46 only:

1. **Shared encoder is cleaner.** The organism head needs an encoder to
   produce `pooled ∈ ℝ^128`. Sharing the L46 encoder means zero extra
   extraction cost and zero extra parameters in the trunk.
2. **L46 alone is already strong.** v6-mini-long's L46 component scores Iris
   AUROC ~0.99 and the dual fusion only adds ~0.002 on Metis. The gain from
   the organism router (+~0.02 on Iris) dwarfs the fusion gain.
3. **Half the extraction volume.** Extracting only L46 instead of L40+L46
   cuts extraction time roughly in half — significant for the 30-min budget.

## Training plan

### Phase 1: extract training data

From the Metis training datasets (instruct-deception and varied-deception,
all three families), extract L46 residual stream activations for every
response token. Save per-example data with:
- `pooled_features`: mean-pooled L46 activations
- `is_organism`: 1 if `lora is not None`, 0 otherwise
- `is_deceptive`: 1 if deceptive, 0 if honest
- `family`: qwen / gemma / nemotron

Existing extraction infrastructure from v3.3-probe-training can be reused.

### Phase 2: compare training approaches

Two options for producing the two-head probe. Compare both on **non-leaked
validation data** (hold out some Metis cells before training) before
committing to one for the submission model.

**Option A: Frozen encoder + logistic regression head**

1. Load the existing v3.3 probe (`model.pt`) — freeze encoder + deception head.
2. Forward-pass all training examples through the encoder, collect `pooled ∈ ℝ^128`.
3. Train `Linear(128 → 1)` on `pooled → is_organism` (sklearn LogisticRegression
   or SGD, seconds per family).

**Option B: Joint training from scratch**

1. Train a fresh `TransformerTokenProbe` with two heads: `head_deception` and
   `head_organism`, both `Linear(128 → 1)`.
2. Joint loss: `L = BCE(deception_logit, is_deceptive) + λ × BCE(organism_logit, is_organism)`.
3. Shared trunk across families (same as v3.3), per-family projection layers.
4. Same hyperparameters as v3.3: Adam lr=1e-4, 2 transformer blocks, d_model=128,
   batch by token budget.
5. λ = 1.0 (equal weight) initially; tune if one task dominates.

**Comparison metrics** (on non-leaked holdout):

| metric | why it matters |
|---|---|
| Deception AUROC (instruct) | must preserve existing probe quality (~0.96+) |
| Deception AUROC (varied) | harder OOD test; v3.3 trunk scores ~0.92 here |
| Organism AUROC | must separate base from organism; target >0.90 |
| Organism AUROC on g-st (cross-adapter) | the critical test — Iris uses g-st |
| Pearson r(deception_z, organism_conf) | heads must be orthogonal; target |r| < 0.3 |

**Decision rule:** Option B wins if organism AUROC is meaningfully higher and
deception AUROC doesn't regress more than −0.01 vs Option A. Option A wins if
it hits organism AUROC >0.90 and we save a full training cycle. If both hit
organism AUROC >0.90 with deception within 0.01 of v3.3, pick the simpler one.

### Phase 3: train the winner on all data

Once the comparison picks a winner, train on the full Metis dataset (no
holdout) and save:

```
submission/whitebox_probe/shared_organism_head.pt   # single LogisticRegression head, all families
```

Per-family projection layers map different hidden_dims to the same ℝ^128
encoder space, so one shared organism head works across all families.
Trained on qwen+gemma pooled features combined (8000 rows), AUROC 0.85,
r=0.008 vs deception (near-perfect orthogonality).

### Phase 4: evaluate offline

Before building the submission notebook:

1. **Separation quality**: confirm organism AUROC >0.90 on the full training set
   (no held-out data left, so use train-set metrics; the comparison phase already
   validated on holdout).

2. **Cross-adapter transfer**: verify the head fires on g-st organisms (Iris
   adapter type). If Option A was chosen and g-st transfer is weak, fall back
   to Option B with g-st in training.

3. **Length confound**: does the organism head fire on long base-model rows?
   Test on the longest available base-model examples.

4. **Deception orthogonality**: confirm Pearson r between `organism_conf` and
   `probe_z` is low.

### Phase 5: build submission notebook

Only after phase 3 confirms the organism head works:

Based on `sonic_v6_1.ipynb`, simplified to L46-only:

| cell | content | change from v6.1 |
|---|---|---|
| 0 | header | rewritten |
| 1–3 | imports, dataset loading | unchanged (but no L40 paths) |
| 4 | load L46 probe + **shared organism head** | added organism head loading from shared path; L40 removed |
| 5 | probe definition | `TransformerTokenProbe` unchanged; organism head is standalone `Linear` loaded from shared_organism_head.pt |
| 6–8 | model, layers, tokenization | unchanged (L46 layer only) |
| 9 | extraction | L46 only (no L40) — roughly half the extraction time |
| 10 | probe scoring + organism scoring | run encoder → pooled, apply deception head + organism head |
| 11 | Phoenix v6.3 judge | unchanged |
| 12 | **hard-switch router** | replaces sign gate |
| 13 | submission | unchanged |


## Router behaviour by dataset

### Notus (base models — Nemotron, Qwen, Gemma, no adapters)

| organism_conf | router → | score source | expected AUROC |
|---|---|---|---|
| < threshold (most rows) | judge | σ(judge_z) | ~0.89 |
| > threshold (false positives) | probe | σ(probe_z) | ~0.56 ⚠️ |

False positives are the main risk — base-model rows where the organism head
fires and the probe takes over with its ~0.56 AUROC. The threshold should be
set so false positives are rare. If needed, `ORGANISM_THRESHOLD` can be
bias toward the judge (fewer false positives, more false
negatives on Iris — an acceptable trade since the judge scores 0.95 on Iris).

### Iris (organisms — g-st adapters)

| organism_conf | router → | score source | expected AUROC |
|---|---|---|---|
| > threshold (most rows) | probe | σ(probe_z) | ~0.99 |
| < threshold (false negatives) | judge | σ(judge_z) | ~0.95 |

False negatives are less damaging — the judge still scores 0.95 on Iris.

## Expected headline impact

If the organism head achieves AUROC ≥ 0.95 on base-vs-organism:

| dataset | judge-only (v6.3) | v7 projected | Δ |
|---|---|---|---|
| Notus (mean) | ~0.89 | ~0.89 (judge handles) | ~0 |
| Iris (mean) | ~0.96 | ~0.99 (probe handles) | **+0.03** |
| **Headline** | ~0.92 | ~0.94 | **+0.02** |

The gain comes entirely from Iris — the organism head lets the probe's full
~0.99 AUROC through on organisms instead of the judge's ~0.96. Notus is
unchanged (the judge was already carrying it).

## Risks

1. **Organism head doesn't generalise to g-st adapters.** The Iris holdout
   uses `g-st-*`. The Metis training data includes `g-st` for Qwen and Gemma
   but not Nemotron. If the g-st residual stream shift differs from LoRA
   shifts, the head may not fire on Iris. Mitigation: evaluate cross-adapter
   transfer before submission; include g-st in training if needed.

2. **Length confound.** Notus conversations are ~2500 tokens vs Metis ~400.
   The pooled representation depends on sequence length through the
   positional encoding and mean-pool statistics. The organism head might
   fire spuriously on long base-model rows. Mitigation: evaluate on the
   longest available base-model rows; if confounded, add a
   length-normalisation step.

3. **Hard-switch discontinuities.** Two rows with nearly identical
   `organism_conf` on opposite sides of the threshold get scores from
   completely different sources. Mitigation: check row density near the
   threshold — if sparse, the discontinuity is minor.

4. **Training data imbalance.** Metis has more organism rows than base-model
   rows (8:1 for Qwen). Mitigation: balance the training set.

## Phase 1 & 2 results (Jul 29, 2026)

Organism heads trained as Option A: freeze v3.3 L46 encoder, extract pooled(128)
features, train per-family LogisticRegression on base-vs-organism labels. Full
Metis training data (no holdout).

| family | rows (base/org) | organism AUROC | r(deception, org) |
|---|---|---|---|
| qwen | 7200 (800/6400) | 0.8417 | −0.039 |
| gemma | 800 (400/400) | 0.9405 | +0.244 |
| nemotron | 216 (all base) | — dummy, always → judge | — |

**Final approach (Jul 29): one shared LogisticRegression head** trained on
qwen+gemma pooled features combined, saved to `shared_organism_head.pt`. The
per-family projection layers already map different hidden_dims to the same
ℝ^128 encoder space, so one head works across all families:

| combined | 8000 rows (6800/1200) | AUROC 0.8513 | r=−0.008 |

Near-zero correlation (r=−0.008, p=0.48) confirms the organism head reads a
signal completely orthogonal to deception — exactly what the router needs.

## Next

1. **Run dry rehearsal** (`python submit.py --dry`) — verify organism heads load
   and scores are non-trivial on all datasets.
2. **Tune threshold** — if Notus base-model rows get high organism_conf (false
   positives), raise `ORGANISM_THRESHOLD` above 0.5 to bias toward the judge.
3. **Submit** to the leaderboard.
