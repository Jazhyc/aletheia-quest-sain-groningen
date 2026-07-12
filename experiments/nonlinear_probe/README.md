# Nonlinear probe experiments (white-box, Idea 1)

Deception detection from model internals: extract residual-stream activations
for every dev dataset once, cache them on disk, then sweep probe
configurations offline. This document records the experimental setup and all
results so far.

## Phase 0 — activation extraction

**Script:** `extract_activations.py` (Slurm fallback: `extract_activations.sh`)

One remote NDIF session per dataset (`--remote`, default host
`https://aletheias.api.ndif.us`), reading the dataset's own `model`/`lora`
columns via `submission/util.py`'s `build_model`. Per example:

- Chat is tokenized with `chat_preprocess` (`--max-len 512`), giving the
  assistant-response span.
- At every 2nd decoder layer (`--layers ::2`) two poolings of the residual
  stream are stored: **mean** over the response span and the **last**
  response token, both float16.
- Batches are packed by length with a `batch_size × padded_len ≤ 2560` token
  budget (NDIF OOM guard), pooled per layer inside the trace, moved to CPU
  (NDIF shards 27B models across GPUs), stacked, and `.save()`d once per
  batch; `tracer.stop()` skips the layers above the deepest read.
- Labels are joined from the sibling `<dataset>-labels` HF repo by `index`.

Output: `results/whitebox/activations/<dataset>.npz` with `mean_L{L}` /
`last_L{L}` arrays of shape (400, hidden), plus `index`, `deceptive`,
`response_tokens`, and a JSON `meta` blob (gitignored, ~200 MB each).

**Coverage (2026-07-12):** all 22 dev datasets, 400 rows each, 200/400
deceptive in every one.

| scenario | base model | organisms |
|---|---|---|
| varied | Qwen3.5-27B (64 layers, d=5120; 32 cached layers) | None, a-mo-1/3/4/5/6/7, b-mo, c-mo |
| instructed | Qwen3.5-27B | same 9 + g-st |
| instructed | gemma-3-27b-it (62 layers, d=5376; 31 cached layers) | None, g-st-2, s-mo |

The varied scenario only exists for Qwen; Nemotron-120B is deferred. Full
extraction ran in ~25 min total from the local machine (roughly 40–60 s of
NDIF compute per dataset).

### Pitfalls hit and fixed on the way

- `decoder_layers` (util.py) never matched Qwen3.5's hybrid attention stack
  (layer 0 has `linear_attn`, not `self_attn`) and its fallback broke under
  PEFT nesting; replaced with a walk down `language_model`/`model` envoys.
- transformers 5.x decoder layers return the hidden tensor directly, so
  `.output[0]` sliced the batch dimension.
- Only values `.save()`d inside the session survive remote execution;
  appending to a pre-session list silently loses everything.
- gemma's residual stream overflows float16 (>65504) from layer ~48 up:
  `inf` values in the three gemma caches, clipped at load time in the sweep.
- One transient NDIF `RemoteException` ("Error submitting request"); a plain
  retry succeeded.

## Phase 1 — linear layer sweep

**Script:** `linear_sweep.py` (tests: `test_linear_sweep.py`)

For every (base model, cached layer, pooling ∈ {mean, last, concat}) a
logistic regression (`StandardScaler` + `LogisticRegression(max_iter=2000,
C=1.0)`, float32) is trained on all datasets of one scenario and evaluated on
all datasets of the other, in both directions:

- **Train set:** all datasets of one scenario, pooled. For Qwen, e.g.
  `varied` = base + 8 organisms, ~3600 rows.
- **Eval set:** all datasets of the *other* scenario, pooled — rows the
  probe never saw during fitting. For Qwen, `instructed` = same organisms +
  `g-st`, ~4000 rows.
- **Direction:** run both ways (train varied/eval instructed, and train
  instructed/eval varied) since Qwen has both scenarios cached.
- **gemma exception:** no `varied` data exists for gemma at all, so there is
  no second scenario to hold out against. It falls back to stratified 5-fold
  CV within the pooled `instructed` datasets instead (marked `cv`): fit on
  4/5 of the rows, predict the held-out 1/5, repeat per fold. This is
  in-distribution (same scenario, same construction) and therefore a weaker
  generalization test than the Qwen directions.

Metrics: AUROC and balanced accuracy at threshold 0.5, overall and per eval
dataset. Features are read lazily per layer from the npz caches (loading all
22 at once OOMs a 16 GB machine).

Outputs: `results/whitebox/linear_sweep/{sweep_results,per_dataset}.csv`
(285 and 2103 rows).

The leaderboard averages balanced accuracy *per dataset* (macro), while the
sweep computes it over all eval rows pooled — but because every dev dataset
has exactly 400 rows with a 200/200 class split, the two are identical
(verified: max difference 0.0000 across all 285 configs; each dataset
contributes the same 200 examples to each class, so pooled per-class recall
is the unweighted mean of the per-dataset recalls). All BA numbers below,
including the figure, can therefore be read as leaderboard-style
macro-averages. AUROC is not decomposable this way and differs slightly
(e.g. 0.891 pooled vs 0.887 macro for L46 concat instructed→varied), but it
is only the tie-break metric. Optional `--wandb` / `--wandb-only` logging; run of
record: <https://wandb.ai/ml_exp/aletheias-quest-whitebox/runs/3ajq95ey>.
Runtime ~45 min on a 15 GB WSL2 box (CPU-only; lbfgs on ~4k×5–10k matrices
near linear separability is the bottleneck).

### Results — Qwen3.5-27B (cross-scenario, the honest evaluation)

![Balanced accuracy by decoder layer, mean pooling: Qwen trained on varied and
tested on instructed peaks ~0.96 in layers 36–48, Qwen trained on instructed
and tested on varied tops out ~0.78 with the same shape, gemma's
within-instructed 5-fold CV sits at 0.91+ from layer 0
upward.](figures/layer_curve.png)

"Trained on X, tested on Y" = the probe is fit on all datasets of scenario X
and scored on all datasets of scenario Y, so each curve measures transfer to
an unseen kind of deception. Every point therefore averages over the test
scenario's datasets — one per model organism; the per-organism spread hiding
inside the instructed→varied curve is broken out in the table further down.
Balanced accuracy is the average of the per-class accuracies (0.5 = chance
regardless of class balance).

Mean balanced accuracy over both train→eval directions:

| layer | pooling | mean BA |
|---|---|---|
| **46** | **concat** | **0.876** |
| 40 | concat | 0.876 |
| 38 | concat | 0.876 |
| 42 | last | 0.874 |
| 46 | mean | 0.869 |

The layer curve (mean pooling) rises from ~0.56 at layer 0 to a broad
plateau of **0.84–0.87 across layers 32–50**, decaying slightly to 0.80 at
layer 62. The deception signal sits in the upper-middle of the network —
`tracer.stop()` after layer ~48 is enough at inference time.

Best config (L46 concat) per direction:

| train → eval | AUROC | BA@0.5 | n_train | n_eval |
|---|---|---|---|---|
| varied → instructed | 0.995 | 0.968 | 3600 | 4000 |
| instructed → varied | 0.891 | 0.784 | 4000 | 3600 |

Strong asymmetry: probes trained on varied deception transfer almost
perfectly to instructed (worst dataset 0.930 BA), but not vice versa.
Per-dataset BA for instructed→varied exposes the weak spot — the `a-mo`
organisms under varied deception:

| eval dataset (varied) | BA | AUROC |
|---|---|---|
| a-mo-1 | 0.638 | 0.769 |
| a-mo-3 | 0.615 | 0.767 |
| a-mo-4 | 0.748 | 0.833 |
| a-mo-5 | 0.835 | 0.954 |
| a-mo-6 | 0.783 | 0.879 |
| a-mo-7 | 0.790 | 0.871 |
| b-mo | 0.895 | 0.975 |
| c-mo | 0.883 | 0.971 |
| None (base) | 0.875 | 0.963 |

### Results — gemma-3-27b-it (5-fold CV within instructed)

Near ceiling everywhere in the mid-stack: best 0.989 BA (L26 mean), with
0.987–0.988 across layers 26–44. Instructed deception is easy to read off
gemma's activations; no cross-scenario test is possible until varied gemma
data exists. Note in the figure that gemma's CV accuracy is already 0.91 at
layer 0 — within-scenario, surface statistics of the embeddings nearly solve
the task, so these CV numbers say "instructed is easy", not "the probe
generalizes".

### Takeaways

- A plain logistic probe on cached activations already reaches **0.876
  cross-scenario BA**, above the team's official black-box score (0.833),
  under a harsher shift than the leaderboard's (a deployed probe can train
  on both scenarios at once).
- Best operating point: **Qwen layers 36–48**, concat(mean, last) pooling.
- The bottleneck is generalization to the `a-mo` organisms on *varied*
  deception (0.615–0.835 BA). This is where Phase 2 (nonlinear probes:
  MLP, attention pooling over per-token activations, multi-layer fusion)
  should aim, evaluated with leave-one-organism-out in addition to
  cross-scenario.

## Phase 2 — nonlinear probes and organism holdout

**Script:** `nonlinear_sweep.py` (tests: `test_nonlinear_sweep.py`)

Compares an MLP probe (`StandardScaler` + `MLPClassifier`, one hidden layer
of 512, Adam, early stopping on a 15% validation split) against the Phase 1
logistic baseline on the same caches, under two protocols:

- **Cross-scenario** (as in Phase 1): layers 36–48, all poolings, both
  directions.
- **Leave-one-organism-out** (new): train on all datasets of *both*
  scenarios except the held-out organism's, evaluate on that organism. This
  is the deployment-realistic protocol — a submitted probe trains on all dev
  data and the hidden datasets may contain unseen organisms — and it is the
  organism-generalization test cross-scenario evaluation cannot provide
  (the same organisms appear in both scenarios).

Outputs: `results/whitebox/nonlinear_sweep/{cross,holdout}_results.csv`
plus per-dataset breakdowns. Runtime ~1 h CPU (~43 s per MLP fit at L46
concat).

![The Phase 1 layer-curve figure with the MLP's cross-scenario results
overlaid as dashed open-marker segments at layers 36–48, mean pooling: the
MLP tracks the logistic curves closely in both
directions.](figures/layer_curve_mlp.png)

Same figure as Phase 1, with the MLP probe (dashed, mean pooling) overlaid
at the layers it was run on. gemma has no MLP curve because it has no
cross-scenario evaluation.

### Cross-scenario: nonlinearity helps only marginally

Best config is again L46 concat; the MLP lifts the mean over both
directions from 0.876 to **0.888**, entirely from the hard
instructed→varied direction (0.806 vs 0.784 BA, 0.903 vs 0.891 AUROC).
The easy direction is saturated (~0.97) for both probes.

### Leave-one-organism-out: the headline result

Training on both scenarios changes the picture completely. Macro BA over
the 10 held-out Qwen organisms (L46 concat):

| probe | mean BA | mean AUROC | worst organism |
|---|---|---|---|
| logistic | 0.914 | 0.978 | a-mo-3 (0.840) |
| mlp-512 | **0.920** | 0.980 | a-mo-1 (0.856) |

Every organism scores ≥ 0.84 even when the probe has never seen it —
compared to 0.615 for a-mo-3 under the Phase 1 instructed→varied protocol.
Phase 1's pessimistic 0.78 plateau was mostly a *scenario* gap (no varied
data in training at all), not an organism gap. Once some varied data from
other organisms is in the training set, unseen-organism generalization
lands at **≈0.92 BA / 0.98 AUROC** — the realistic deployment estimate,
well above the black-box baseline (0.833). gemma holdout stays near
ceiling (0.95–0.99 across its 3 organisms).

The residual weak spot is unchanged in kind but smaller: the varied-scenario
datasets of a-mo-1/3/4 (0.68–0.79 BA per dataset, while their instructed
counterparts are ≥ 0.98). The MLP's overall gain is small (+0.005 macro)
and not uniform — it wins on a-mo-3/5/6/7, b-mo, g-st and loses slightly
on None, a-mo-1, a-mo-4, c-mo.

### Takeaways

- Probe capacity is **not** the bottleneck: a 512-unit MLP buys ≈0.005–0.012
  BA over logistic. Deeper/wider probes are unlikely to close the a-mo
  varied gap; richer features (per-token attention pooling, multi-layer
  fusion) would be the next lever if we chase it.
- The deployment-realistic score estimate is **≈0.92 macro BA** (train on
  everything, unseen organisms at eval), not the 0.876 cross-scenario
  number.
- Either probe at **L46 concat** is submission-ready; the MLP is slightly
  better on the weakest organisms, the logistic is simpler and more
  predictable.

### Token-level probes (per-token attention / CNN / transformer)

**Script:** `token_sweep.py` (probe architectures in `token_probes.py`,
tests: `test_token_sweep.py`)

Instead of a single pooled vector per example (mean or last token), per-token
probes consume the full `(tokens, hidden)` sequence of response-token
activations at one layer, letting them learn *which* tokens carry the
deception signal:

- **AttentionTokenProbe:** token projection → learned attention pooling
  (masked softmax) → linear head.
- **CNNTokenProbe:** two `Conv1d(k=5, padding="same")` layers over the token
  axis with masked mean pooling.
- **TransformerTokenProbe:** projection + sinusoidal position encoding →
  *n*×*n* self-attention (masked via `src_key_padding_mask`) → masked mean
  pooling.

Three architectures at **layers 40 and 46** (the two candidate layers from
the extraction sweep, chosen post-hoc because the layer curve peaks in the
36–48 band), using the same cross-scenario and leave-one-organism-out
protocols as the MLP sweep. Training uses Adam, early stopping on a 15%
validation split, and variable-length batch packing with a token budget of
8192 to stay within GPU memory.

**Token extraction:** `extract_activations.py --tokens` writes per-token
activations for every response token at layers 40 and 46 into
`<dataset>.tokens.npz` (~14 GB total across all 22 datasets, 5376-d for
gemma, 5120-d for Qwen). The pooled caches (mean/last) remain available for
the logistic and MLP probes.

Outputs: `results/whitebox/token_sweep/{cross,holdout}_results.csv`.
Runtime ~9 min GPU (RTX 4090), dominated by transformer encoder sweeps.

### Cross-scenario (token probes)

| architecture | instructed→varied (BA) | varied→instructed (BA) | mean BA |
|---|---|---|---|
| attention L46 | 0.785 | 0.920 | 0.852 |
| cnn L46 | **0.789** | 0.958 | **0.873** |
| transformer L46 | 0.782 | **0.962** | 0.872 |
| attention L40 | 0.758 | 0.911 | 0.834 |
| cnn L40 | 0.778 | 0.960 | 0.869 |
| transformer L40 | 0.759 | 0.958 | 0.858 |

The scenario gap persists: instructed→varied BA remains ~0.79 regardless of
architecture, matching the logistic/L46 concat baseline (0.784). Token-level
probes are not the lever that closes this gap.

### Leave-one-organism-out (token probes)

Mean BA across all held-out Qwen organisms, compared to the L46 concat
logistic and MLP baselines (from the nonlinear sweep):

| probe | mean BA | mean AUROC | worst organism BA |
|---|---|---|---|
| logistic (L46 concat) | 0.915 | 0.977 | 0.840 (a-mo-3) |
| mlp-512 (L46 concat) | 0.920 | 0.979 | 0.855 (a-mo-3) |
| **transformer token L46** | **0.923** | 0.977 | **0.859 (a-mo-1)** |
| cnn token L46 | 0.911 | 0.975 | 0.859 (a-mo-1) |
| attention token L46 | 0.879 | 0.971 | 0.672 (g-st) |
| transformer token L40 | 0.912 | 0.976 | 0.865 (a-mo-1) |
| cnn token L40 | 0.904 | 0.975 | 0.849 (a-mo-3) |
| attention token L40 | 0.897 | 0.974 | 0.795 (g-st) |

**The transformer token probe at L46 is the best single configuration.**
It has the highest mean holdout BA (0.923) and the highest worst-case BA
(0.859) — no other probe beats both numbers. It wins decisively on the
hardest organisms:

| organism | transformer token L46 | best non-token (logistic/MLP, L46 concat) |
|---|---|---|
| a-mo-1 | 0.859 | **0.865** (logistic) |
| a-mo-3 | **0.889** | 0.855 (MLP) |
| a-mo-4 | **0.890** | 0.886 (logistic) |
| a-mo-5 | **0.948** | 0.932 (MLP) |
| a-mo-6 | **0.917** | 0.910 (MLP) |
| a-mo-7 | **0.917** | 0.910 (MLP) |
| b-mo | 0.895 | **0.959** (MLP) |
| c-mo | 0.958 | **0.973** (logistic) |
| g-st | **0.995** | 0.977 (MLP) |
| None (base) | 0.964 | **0.966** (logistic) |

The token probe improves on 5 of the 10 organisms (a-mo-3/4/5/6/7, g-st),
while the logistic/MLP wins on the other 5 (a-mo-1, b-mo, c-mo, None).
Notably a-mo-3 jumps from 0.855 (MLP) to 0.889. The oracle upper bound
(picking the best probe per organism) is 0.932 — only 0.009 above the
transformer alone, leaving essentially no headroom from architecture
improvements.

The one striking failure is **attention pooling at L46 on g-st** (0.672 BA,
the worst score across all probes tested), though both CNN and transformer
handle it fine (0.887 and 0.995 respectively) — attention pooling alone is
brittle when the learned attention pattern mismatches the token distribution
of an unseen organism.

Score-level ensembles (logistic + transformer, averaging predictions) do not
help — the logistic is too weak on the hard organisms and drags the average
down (mean 0.919, worst 0.850). Two-layer token fusion (concatenating L40
and L46 token representations) also underperforms the single-layer
transformer (mean 0.916, worst 0.855). Neither the scenario gap nor the
organism gap can be closed by probe architecture alone.

**Pitfall:** gemma's L46 token caches contain ~700 `inf` values at two
columns (float16 overflow, matching the pooled-cache overflow seen in Phase
1). A `.clamp()` in `load_token_features` clips them; the clamp was added
after the initial sweep crashed with NaN scores.

### Takeaways (Phase 2)

- Probe capacity is **not** the bottleneck: a 512-unit MLP buys ≈0.005–0.012
  BA over logistic; token-level probes (transformer, CNN) improve the hard
  organisms but leave the scenario gap untouched.
- **The best single configuration is the transformer token probe at L46**
  (0.923 mean holdout BA, worst case 0.859). It is GPU-intensive but
  outperforms the simpler logistic/MLP baselines on the hardest organisms.
  The deployment notebook will use this probe for now.
- The deployment-realistic score estimate is **≈0.92 macro BA** (train on
  everything, unseen organisms at eval), not the 0.876 cross-scenario
  number — and the leaderboard gap to the black-box baseline (0.833) is
  substantial.
- Next: per-dataset threshold calibration and the deployment notebook (Phase
  3).

## Reproduction

```bash
# one dataset, remote extraction (needs NDIF_API_KEY and gated HF access)
python experiments/nonlinear_probe/extract_activations.py --remote \
    --dataset aletheias-quest/dev-varied-deception-Qwen3.5-27B-None

# sweep over all cached datasets, then upload to wandb
python experiments/nonlinear_probe/linear_sweep.py
python experiments/nonlinear_probe/linear_sweep.py --wandb-only --wandb-entity ml_exp

# Phase 2a: MLP vs logistic, cross-scenario + leave-one-organism-out
python experiments/nonlinear_probe/nonlinear_sweep.py

# Phase 2b: token-level probes (attention, cnn, transformer) — GPU needed
python experiments/nonlinear_probe/token_sweep.py

# regenerate the layer-curve figure (writes results/whitebox/linear_sweep/,
# copy into experiments/nonlinear_probe/figures/ to refresh the README)
python experiments/nonlinear_probe/plot_layer_curve.py
```

Environment: nnsight 0.7.1.dev, transformers 5.12.1, scikit-learn 1.9.0,
torch 2.11.0, numpy 2.2.6 (local `.venv`, Python 3.12).
