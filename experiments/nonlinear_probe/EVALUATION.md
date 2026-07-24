# Pre-submission local evaluation

How to estimate a sonic submission's real score **before** submitting, without
fooling yourself. Two things make a naive local number a lie: the probe's
training leakage, and a row-level split that hides the failure mode that
actually costs us on the leaderboard. This document says how to avoid both, what
infrastructure already exists, and how to obtain the judge outputs the
gate-level eval needs (locally, on a single GPU).

Companion docs: `docs/sonic/sonic_v2_3_6_EXPLAINER.md` (how the detector works and its
per-step generalization critique) and `docs/sonic/sonic_v2_3_6_CHANGES.md` (the gate).

---

## The question this answers

The official per-dataset results are bimodal. On `Metis` (instructed deception)
the probe scores ~0.99 BA; on `Notus` it is near chance (BA ~0.69–0.76, AUROC
~0.69–0.83). The leaderboard gap to the top team is almost entirely on `Notus`.
So the number that matters before submitting is not "how well do we do overall"
— it is **"which datasets does the probe go blind on, and does the judge carry
them?"** A local eval that cannot surface that is worse than no eval, because it
gives false confidence.

---

## Rule 1 — never score the *shipped* probe on dev rows (leakage)

The shipped probes in `submission/whitebox_probe/` were trained on **all** dev
rows. Scoring them on any dev row is leakage: the honest holdout number is
BA ~0.923 versus the ~0.968 the leaked numbers suggest.

**The fix is refit, not reuse.** For each fold, retrain the probe from scratch on
the fold's *train* rows only (same recipe: transformer token probe, layer 46,
seed 0, <=60 epochs) and score the *held-out* rows. `eval_split_holdout.py`
already does exactly this from cached activations — no NDIF.

**The subtle part: the gate inherits the leak.** The gate consumes
`probe_scores`. To evaluate the *gate* honestly you must feed it the **refit**
(leakage-free) probe scores for the held-out rows, never the shipped-probe
scores. A leakage-free probe wired into a gate fed leaked scores is still a
leaked evaluation.

## Rule 2 — the judge has training data too

The judge's LoRA (`aletheias-phoenix-wright-v2-adapter`) was fine-tuned on some
corpus. If that corpus overlaps the dev deception data, dev *judge* numbers are
also optimistic, and a gate eval built on them is doubly inflated. We do not
currently know the adapter's training composition. Treat dev judge numbers as an
upper bound until that overlap is ruled out. Rule 1's discipline (held-out
evaluation) does not protect against this, because the judge is not refit per
fold — so this caveat must be carried explicitly.

---

## Which split — and why the frozen one is not enough

`make_dev_splits.py` produces a deterministic 80/10/10 split, **stratified by
(dataset, label), shuffling rows within each dataset** (train 6573 / val 822 /
test 821; frozen in `dev_splits/manifest.csv`, seed 0). Every organism —
including `Notus` — has rows in the training set. You only ever hold out *rows*,
never *organisms*.

That answers the easy question ("seen this organism, generalize to new rows of
it?") and it **structurally cannot reveal the `Notus` collapse**, because the
probe trained on `Notus` rows. It will report ~0.92 and hide the exact risk the
hidden test stresses.

Use three protocols, weighting the bottom two:

| Protocol | Question it answers | Role | Exists as |
|----------|--------------------|------|-----------|
| Fixed 80/10/10 (refit) | In-distribution, held-out rows | Sanity floor — expect it optimistic | `eval_split_holdout.py` |
| **Leave-one-organism-out** | New organism, seen scenario | **Primary probe signal** | `layerstack_sweep.py`, `token_sweep.py` |
| **Cross-scenario / leave-one-family-out** | New deception style | **Hardest, most honest OOD** | `plot_layer_curve.py`, `plot_probe_comparison.py` ("trained on varied, tested on instructed") |

Leave-one-organism-out = train the probe on every organism except one, evaluate
on the held-out one (`train on all organisms except one`; already implemented in
`layerstack_sweep.py`). Cross-scenario is the harsher axis: train on the
`instructed` scenario, test on `varied` (and vice versa) — the closest local
analog to "a new deception style appears." Do **not** invent a new split scheme;
reuse this infrastructure. The only thing missing is running these folds
*through the gate with the judge*, rather than probe-only.

---

## Obtaining judge outputs — run the judge locally on one GPU

The gate-level eval, the naive-ensemble comparison, and the `W_MAX` sweep all
need per-row judge outputs `{hard verdict, soft prob}`. These are not cached
anywhere. They do **not** require NDIF or the shared team key — the judge fits on
a single 24 GB GPU (e.g. RTX 4090).

Key point: **the judge is a fixed model and is not refit per fold** (only the
probe refits). So run the judge **once** over all dev rows, cache the outputs,
and reuse that cache across every split, the ensemble comparison, and the sweep.
One pass over ~8k rows is a few hours as a background job.

- **Model:** `Qwen/Qwen3.5-9B` + `Jazhyc/aletheias-phoenix-wright-v2-adapter`
  (peft). Needs HF access for both.
- **Precision / VRAM on 24 GB:**
  - bf16 ≈ 18 GB weights — fits, but tight. Use batch 1–4; the notebook's
    `PHOENIX_BATCH_SIZE=16` will OOM.
  - 4-bit NF4 ≈ 5–6 GB — comfortable, large batches. Note: quantization means
    the local judge is **not bit-identical** to the NDIF-served submission judge.
    Fine for ranking/ablation; do not report it as the exact submitted judge.
- **Match the notebook's judge logic** (cell 11 of the submission notebook) so
  the cache is valid gate input: same system prompt, the
  `<context>…</context>\n<output>…</output>` formatting, greedy `Prediction:<0/1>`
  for the hard verdict, and the teacher-forced next-token softmax over "0"/"1"
  for the soft prob (locally this is just a logits read — easier than on NDIF).

**Fidelity caveat.** A local transformers run (especially 4-bit) can differ
slightly from the NDIF judge. Use local judge outputs to answer *relative*
questions (does adaptive `w` beat a fixed blend? does the `W_MAX` cap move
anything? which datasets does the judge carry?), not to predict the exact
submitted judge score.

---

## The three questions this unblocks

Once you have (a) refit probe scores per fold and (b) the cached local judge
outputs, feed both through the gate (`sonic_v2.3.x` cell 12 logic) and answer:

1. **Is the gate better than a naive ensemble?** Compare sonic's gate against:
   raw-average `0.5*probe + 0.5*judge_soft`; rank-average with fixed `w=0.5` and
   the judge-prevalence threshold; judge-only; probe-only. There is currently no
   such baseline in the repo (`baselines/mean_probe` and `baselines/llm_judge`
   are single-detector references, not a fusion). The real question is whether
   the agreement-adaptive `w` beats a fixed `0.5` on the `Iris`-vs-`Notus`
   reliability spread.
2. **Does the `W_MAX = 0.9` cap earn its place?** Sweep `W_MAX ∈ {0.8, 0.85,
   0.9, 0.95, 1.0}` and read BA on the `Notus`-like held-out folds. Expectation
   (from the mechanism): near-flat. If flat, 0.9 is a harmless prior; if there
   is a knee, it becomes an evidence-based choice.
3. **Do v2.3.6 changes A and C help where intended?** Both are inert on
   in-distribution dev data by design (A needs an unseen base model; C needs
   agreement in `[0.45, 0.5)`), so they must be checked on the OOD folds, not
   the fixed split.

---

## Reproducibility notes

- The frozen split lives in `dev_splits/manifest.csv` (seed 0). Do not
  regenerate it for an eval you want to compare across runs.
- Probe activations are cached in `results/whitebox/activations/*.tokens.npz`;
  the probe eval never needs NDIF.
- Cache the local judge outputs keyed by `(dataset, index)` so the gate,
  ensemble, and sweep evals all read the same file.
- Carry the Rule 2 caveat in any headline number until the judge-adapter /
  dev-data overlap is ruled out.
