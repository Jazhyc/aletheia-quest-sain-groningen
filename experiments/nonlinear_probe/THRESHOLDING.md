# Thresholding the white-box probes

Why deployment thresholds every probe at **0.5**, why tuned thresholds were
evaluated and rejected, and why nemotron's `0.0672` is an artefact rather
than a counter-example.

Context: `README.md` (Phase 3 — deployment). Deployment notebook:
`submission/sonic_v1.ipynb`.

## Use 0.5, do not tune

Balanced accuracy — the leaderboard's primary metric — depends on the
decision threshold, so it is worth being deliberate here. **Deployment uses
0.5 for every probe.**

Per-dataset optimal thresholds, estimated via 5-fold stratified CV on the
Qwen dev data using out-of-fold scores from the transformer token L46 probe:

| dataset | BA@0.5 | BA@optimal | optimal threshold |
|---|---|---|---|
| instructed (10 datasets) | 0.987–1.000 | 0.990–1.000 | 0.33–0.94 |
| varied None | 0.925 | **0.935** | 0.77 |
| varied a-mo-1 | 0.733 | **0.740** | 0.25 |
| varied a-mo-3 | 0.738 | **0.762** | 0.13 |
| varied a-mo-4 | 0.808 | **0.820** | 0.26 |
| varied a-mo-5 | 0.910 | **0.912** | 0.49 |
| varied a-mo-6 | 0.835 | **0.842** | 0.52 |
| varied a-mo-7 | 0.830 | **0.840** | 0.58 |
| varied b-mo | 0.905 | **0.912** | 0.61 |
| varied c-mo | 0.940 | **0.950** | 0.71 |

This table is the argument *against* tuning. Per-dataset optima scatter over
0.13–0.94 with no stable centre, and buy only 0.922 → **0.929** macro BA —
and they are unusable anyway, since the graded datasets are hidden and we
cannot fit a threshold per dataset we have never seen. The *global* optimum
is 0.50–0.60 (BA 0.922), i.e. 0.5 is already at the optimum to within noise.
Every dev set here is 400 rows at a 200/200 split, and balanced accuracy
weights both classes equally, so a probe trained on balanced data has no
reason to want a boundary anywhere else.

## The nemotron exception (and why it is not a counter-example)

`nemotron_probe/config.json` carries `"threshold": 0.0672`, exported by
`nemotron_probe_train.py`. It is the balanced-accuracy `argmax` over the
unique out-of-fold scores. It is *not* evidence that low thresholds are
right; it is an artefact of the nemotron dev set, which is unlike every
other dataset in this collection:

| | rows | positives | prevalence |
|---|---|---|---|
| every qwen / gemma dev set | 400 | 200 | 0.500 |
| `dev-instructed-deception-…Nemotron-3-Super-120B…` | 216 | 16 | **0.074** |

The 216 rows are the real dataset size (confirmed against the HF dataset,
not a truncated extraction). Training on 7.4% positives squashes the probe's
outputs toward zero, so the BA optimum is dragged down to roughly the base
rate — 0.0672 ≈ 0.074 is the tell.

That number is fragile for three reasons:

1. **It rests on 16 positive examples.** 5-fold CV leaves ~3 held-out
   positives per fold; an `argmax` over their unique scores carries three
   digits of false precision.
2. **It was measured on a model we do not ship.** The threshold comes from
   the fold models' out-of-fold scores, but the script then refits the probe
   on 100% of the data and exports *that* — a differently-scaled model.
3. **It treats the symptom.** Balanced accuracy weights classes equally, so
   the optimal boundary should not track the class prior at all. It only
   does so here because imbalanced training skewed the score distribution.

### Measured: the tuned threshold was *worse* than 0.5

Scoring the **shipped** nemotron weights (`model.pt`, the full-data refit)
over its own token cache settles it. Reason 2 above is not hypothetical —
the refit probe separates far more sharply than the fold models the
threshold was fitted to, and its scores are bimodal, not squashed:

```
score distribution: min=0.0012  median=0.0014  max=0.9972
AUROC        = 0.9988
BA @ 0.5     = 0.9663   (predicts 16 deceptive; 16 true positives)
BA @ 0.0672  = 0.9612   (predicts 18)
```

`0.0672` costs ~0.005 BA against the naive 0.5 on the very model we ship. It
was never a calibration — it was a cutoff fitted to a *different* model's
score scale. (These figures are in-sample, so the absolute values are
optimistic; the threshold-vs-threshold comparison on one fixed model is the
part that matters.)

**Resolution:** the `threshold` key was removed from
`nemotron_probe/config.json`. All three probes now threshold at 0.5 via the
notebook's `config.get("threshold", 0.5)` fallback — one method, one cutoff,
nothing per-organism to justify.

A class-balanced retrain of the nemotron probe is therefore **no longer
needed for thresholding** — the refit probe already lands its boundary near
0.5 unaided. It may still be worth doing on the merits (16 positives is a
thin training signal, and nemotron was the weakest cell of the black-box
run), but that is a probe-quality question, not a threshold one.

Caveat: there is only one nemotron dev set, so no transfer of *any* nemotron
threshold can be measured. That is a further argument for the least-tuned
option.
