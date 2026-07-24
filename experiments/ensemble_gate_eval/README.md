# ensemble_gate_eval

Does the sonic gate earn its complexity over a naive ensemble? This experiment
scores the sonic v2.3.6 gate against fixed-weight and single-detector baselines
on the same leakage-free, out-of-distribution folds, so the comparison is
attributable rather than a prior.

Read `experiments/nonlinear_probe/EVALUATION.md` first — it defines the leakage
rules, the split protocols, and how to obtain judge outputs locally. This folder
is the harness that implements the comparison described there.

## Design

The experiment reuses the submission notebook's **gate logic**, but does not
execute the notebook (that path uses the leaked shipped probe and is NDIF- and
per-dataset-shaped). Instead:

- `gate.py` — `run_gate` is a faithful lift of cell 12 of
  `submission/sonic_v2.3.6.ipynb`, plus the baselines (`naive_rank_avg`,
  `raw_avg`, `judge_only`, `probe_only`). Every method shares the gate's rank
  transform and prevalence threshold, so the only moving part versus `run_gate`
  is the blend weight — any score difference is attributable to the adaptive
  weight, not to preprocessing.
- `test_gate_parity.py` — executes cell 12's real source and asserts it matches
  `run_gate` on 400+ random and crafted scenarios. This pins `gate.py` to the
  submitted gate: if cell 12 changes, the test fails until `gate.py` is resynced.
  **Run this first; it needs no GPU and no data.**

The two ingredients the harness assembles around the shared gate (neither from
the notebook):

- **Refit probe scores**, per fold, leakage-free — retrain the probe on the
  fold's train rows only and score the held-out rows, from cached activations
  (`results/whitebox/activations/*.tokens.npz`; recipe in
  `experiments/nonlinear_probe/token_probes.py` / `eval_split_holdout.py`). Never
  the shipped-probe scores.
- **Judge outputs** — `{verdict, soft_prob}` per row from `Qwen3.5-9B` +
  `aletheias-phoenix-wright-v2-adapter`, run **once** on a local GPU (the judge
  is fixed, not refit per fold) and cached keyed by `(dataset, index)`.

## Status

- [x] `gate.py` — sonic gate + baselines
- [x] `test_gate_parity.py` — pins `gate.py` to notebook cell 12 (404 scenarios)
- [x] `run_judge_local.py` — local judge runner -> `judge_cache.json` (logic +
      deps verified; the full pass is the user's to run on the 4090)
- [x] `run_eval.py` — LOO / cross-scenario harness: refit probe per fold, read
      judge cache, score every method, print the per-fold + aggregate table
      (probe side verified end-to-end on GPU)

## Run order

```bash
# 1. Prove the eval scores the exact submitted gate (no GPU, no data):
python experiments/ensemble_gate_eval/test_gate_parity.py

# 2. Build the judge cache ONCE on the GPU (hours; resumable, checkpoints per
#    dataset). bf16 needs no extra dep; --load-4bit needs `pip install
#    bitsandbytes` and is lighter on VRAM. Smoke it first with --limit:
python experiments/ensemble_gate_eval/run_judge_local.py --limit 4 \
    --datasets aletheias-quest/dev-instructed-deception-Qwen3.5-27B-None
python experiments/ensemble_gate_eval/run_judge_local.py           # full pass

# 3. Score sonic vs the baselines on the OOD folds (GPU, ~tens of min):
python experiments/ensemble_gate_eval/run_eval.py                  # full
python experiments/ensemble_gate_eval/run_eval.py --base-models qwen \
    --max-epochs 8 --limit-organisms 4                             # fast smoke
```

Without step 2, `run_eval.py` still runs but scores only `probe_only` (the
judge-dependent methods are skipped).

## What the harness will answer

Per held-out organism/scenario fold: AUROC and balanced accuracy for each
method, aggregated the way the leaderboard aggregates (per-dataset, then meaned
over validation datasets), plus a per-dataset breakdown mapped to the
`Metis/Eunomia/Notus/Iris` families.

Falsifiable expectation: sonic ≈ `naive_rank_avg` on the easy families and
sonic > `naive_rank_avg` on `Notus` (adaptive `w -> 0` drops the blind probe
that a fixed `w = 0.5` would drag in). If `naive_rank_avg` matches sonic
everywhere, the adaptive gate is complexity without payoff. If sonic *loses* on a
small-sample `Notus` fold, that is evidence for the parked change B (small-sample
weight shrinkage).

The same harness extends to the `W_MAX` sweep (add capped variants as methods)
and the v2.3.6 A/C checks.

## Fidelity note

A local (especially 4-bit) judge is not bit-identical to the NDIF submission
judge. Use these numbers for *relative* questions (adaptive vs fixed weight, cap
sweep, which datasets the judge carries), not to predict the exact submitted
score.
