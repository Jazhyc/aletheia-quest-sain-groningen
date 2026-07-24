# Contrastive probe — leakage-free LOO evaluation

Run: `experiments/contrastive_probe/run_loo_eval.py --base-models qwen gemma
--max-epochs 60` (seed 0), 2026-07-24. Raw output:
`results/contrastive/loo_eval.json`, log `logs/contrastive_loo_eval.log`.

This was the go/no-go for **sonic v3 change 2**. Both probes are refit per fold
on train rows only and scored on a held-out organism, so every number is
leakage-free. The headline metric is the **worst** held-out organism, not the
mean — a probe that averages well while collapsing on one organism is the probe
we already have.

## Results

**qwen — 18 organisms, 7,200 paired rows** (`results/contrastive/loo_eval_qwen_all.json`)

| method | mean AUROC | worst AUROC | worst organism |
| --- | ---: | ---: | --- |
| absolute | 0.9615 | 0.8345 | `varied/a-mo-qwen3.5-27b-1` |
| contrastive | 0.9570 | 0.8401 | `varied/a-mo-qwen3.5-27b-1` |
| **fused** (50/50 rank) | **0.9650** | **0.8507** | `varied/a-mo-qwen3.5-27b-1` |
| invariant (gradient reversal) | 0.9609 | 0.8357 | `varied/a-mo-qwen3.5-27b-3` |

**Noise floor.** The absolute probe was fit twice at seed 0 across two runs and
moved `0.9617 -> 0.9615` mean, `0.8343 -> 0.8345` worst (cuDNN nondeterminism).
Treat differences below **~0.002** as noise.

**gemma — 2 organisms, 800 paired rows**

| method | mean AUROC | worst AUROC |
| --- | ---: | ---: |
| absolute | 0.9902 | 0.9877 |
| contrastive | 0.9971 | 0.9952 |
| fused | 0.9950 | 0.9927 |

gemma is uninformative: only two organisms and both sit above `0.987`, at the
ceiling. Treat qwen as the result.

The instructed-deception folds are saturated (`0.9967--1.0000` for both probes).
All discrimination between methods happens on the seven `varied/*` LoRA folds.

## What the numbers say

**1. The contrastive probe does not replace the absolute probe.** On qwen it is
*worse* on the mean (`-0.0061`) and statistically tied on the worst organism
(`+0.0022`). It is not a better detector; it is an equally good one.

**2. The two feature sets are genuinely decorrelated.** Measured on
`varied/a-mo-qwen3.5-27b-1`: mean cosine between the diff and the absolute
activations is **0.0665**, with `||diff|| / ||absolute|| = 0.168`. So the diff is
close to orthogonal to what the absolute probe reads — the near-identical
per-fold AUROCs are two different representations carrying about the same amount
of deception signal, not one representation read twice.

**3. Fusion therefore helps, but only a little.** Best worst-organism number of
the three (`0.8472`, `+0.0129` over absolute) and best mean (`+0.0028`). This is
the expected payoff shape for averaging decorrelated, equally-strong predictors:
variance reduction, not new capability.

**4. The organism-invariant probe does nothing.** Gradient reversal moves the
mean by `-0.0006` and the worst organism by `+0.0012` — both inside the noise
floor. Per-fold on the nine `varied/*` folds it beats the plain probe on four
and loses on five, every delta under `0.008`. The adversary trains (17 training
organisms per qwen fold, ramped Ganin schedule, model selection on the deception
loss alone), it simply buys no transfer.

The most likely reason is that it attacks the wrong shortcut. Gradient reversal
removes features that identify *which training organism* a row came from. The
dev organisms differ by LoRA adapter; Notus differs by having no adapter at all
and a different deception style. Making the representation invariant to the
former does not make it invariant to the latter.

## Verdict: change 2 does not clear the bar

v3 needs **+0.138** Notus AUROC. Fusion delivers **+0.013** worst-organism
locally — an order of magnitude short — and costs a second forward pass per
batch (2x activation download against a ~970s/1800s budget). That is a bad trade.

**Do not build sonic v3 around the contrastive probe.**

## The bigger finding: the local proxy cannot see the Notus regime

The worst organism we can construct locally scores `0.8343`. Our *best* Notus
unit on the real leaderboard scores `0.8259`. **Every local LOO fold is easier
than every Notus unit.**

So the harness cannot validate a large Notus gain, because it contains no fold
where the probe fails the way it fails on Notus. It can only say what it said
here: on every organism dev can express, the absolute probe is already at
`0.96` mean, and contrastive features add variance reduction on top.

Two consequences:

- This is **not** proof the contrastive probe fails on Notus. It is proof we
  have no local evidence that it succeeds, and that its mechanism (averaging
  decorrelated equal-strength features) is not the kind that closes a `0.138`
  gap.
- It equally undercuts **change 3**, whose selection metric was worst-organism
  LOO AUROC. Selecting a probe against a proxy whose hardest fold is easier than
  our easiest real failure is weakly informative at best — and change 3 duly
  returned a null result, which this proxy cannot distinguish from "the method
  works but the proxy cannot see it".

Notus differs from dev along an axis dev cannot express. Until we have a local
fold that reproduces a sub-0.83 probe, offline probe selection cannot be trusted
to predict Notus transfer.

## Reproduce

```bash
.venv/bin/python experiments/contrastive_probe/run_loo_eval.py \
    --base-models qwen gemma --max-epochs 60
```

Roughly 7 minutes to build the aligned qwen feature set (both caches held in
RAM, ~13 GB peak), then ~20 s per fold.

## Notes for anyone reusing this code

- `experiments/contrastive_probe/train.py` has an **offset-concatenation bug**:
  it prepends a `0` boundary to every file's offsets before concatenating, which
  injects a spurious row boundary between datasets. `run_loo_eval.py` does not
  use it — it builds its own aligned loader. Fix `train.py` before using it.
- The gemma whitebox cache contains ~694 `inf` values (fp16 overflow at
  extraction). They must be clamped before use or probe scores become `NaN`.
  Both `run_eval.py` and `run_loo_eval.py` clamp to the fp16 range.
- Rows are intersected on `(dataset, index)`; a row missing from either cache is
  dropped from both. qwen pairs 7,200 of 7,200; gemma pairs 800.
