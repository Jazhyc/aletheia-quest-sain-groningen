# Q397 generated-reasoning OOD transfer

This experiment asks whether the validation-selected generated-reasoning
readout transfers to public Liars' Bench response modes. It deliberately does
not use Phoenix's HP-KR or action-report routers or specialist prompts.

The frozen protocol applies the same ordinary optimized-Q397 judge to every
row:

- binary-prompt direct `Prediction:` logits;
- a deterministic compact reasoning summary capped at 192 tokens;
- logits after the generated summary; and
- the already-selected 60% direct / 40% reasoning log-odds blend.

It takes 100 rows per category from the existing frozen 400-row category
sample, stratifying by label and source model. The eight categories are
Alpaca, Convincing Game, Gender Secret, Harm Pressure Choice, Harm Pressure
Knowledge Report, Insider Trading, Instructed Deception, and Soft Trigger.
Alpaca is all honest and is retained as a score/FPR control rather than an
AUROC unit.

Run:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/q397_reasoning_ood/run_ndif.py
PYTHONPATH=. .venv/bin/python \
  experiments/q397_reasoning_ood/score_structural_direct_ndif.py
PYTHONPATH=. .venv/bin/python \
  experiments/q397_reasoning_ood/run_structural_reasoning_ndif.py
PYTHONPATH=. .venv/bin/python \
  -m experiments.q397_reasoning_ood.analyze
PYTHONPATH=. .venv/bin/python \
  experiments/q397_reasoning_ood/analyze_token_length.py
```

Ignored artifacts are written to
`results/blackbox/q397_reasoning_ood_transfer_ndif_v1/`.

## Result

The primary comparison uses Phoenix 7's current structural renderer for both
direct and generated-reasoning arms. Across the seven two-class categories:

| category | direct | reasoning | 60/40 blend | blend delta |
| --- | ---: | ---: | ---: | ---: |
| Convincing Game | `0.9678` | `0.9826` | `0.9760` | `+0.0082` |
| Gender Secret | `0.4972` | `0.6030` | `0.5276` | `+0.0304` |
| Harm Pressure Choice | `0.8283` | `0.7593` | `0.8155` | `-0.0128` |
| Harm Pressure Knowledge Report | `0.8733` | `0.6455` | `0.7755` | `-0.0978` |
| Insider Trading | `0.4812` | `0.7552` | `0.6438` | `+0.1627` |
| Instructed Deception | `0.9996` | `0.9906` | `0.9992` | `-0.0004` |
| Soft Trigger | `0.9052` | `0.9215` | `0.9345` | `+0.0292` |
| **macro** | **`0.79324`** | **`0.80823`** | **`0.81031`** | **`+0.01707`** |

The blend won four categories and lost three. A paired stratified row
bootstrap gave a 95% interval of `[-0.00197, 0.03648]` for the macro delta
and a `0.960` probability of a positive delta conditional on these seven
categories. Resampling whole category deltas gave a much wider
`[-0.03230, 0.07635]` interval, reflecting uncertainty about the composition
of unseen OOD modes.

The aggregate signal is not a broad uniform gain. Removing Insider Trading
changes the mean delta over the other six categories to `-0.00720`. On the
five categories not already represented by Phoenix's action and HP-KR routes,
the mean delta is `+0.01093`, but Harm Pressure Choice still loses `0.0128`.
The all-honest Alpaca control retained a 2% fraction above `0.5` for direct,
reasoning, and blend; its mean score fell from `0.0367` direct to `0.0203`
blended.

Matched structural generation took `482.0s`, and post-reasoning scoring took
`39.6s`; the separate direct control took `34.0s`. Forty-five of 800
reasoning replies did not reach a terminal prediction within 192 tokens.
A post-hoc fallback-to-direct rule for missing boundaries did not improve the
earlier matched-tail comparison and is not selected.

For completeness, replaying the exact tail-rendered validation prompt pair
gave a superficially larger blend gain (`0.69361 -> 0.72406`), but this was
mostly recovery from a much weaker OOD direct renderer and is not the
deployment-relevant comparison.

Conclusion: generic reasoning does transfer strongly to action-report
deception and modestly to some other OOD modes, but it is not safe as a
uniform replacement. It especially fails epistemic knowledge reports. Keep
the current direct and specialist paths; this sample supports reasoning only
as a future label-free routed auxiliary, not a global submission change.

## Token-length follow-up

Exact reasoning-generation inputs ranged from 220 to 1,385 tokens (median
458, p95 1,050). Token count is not a stable routing feature. Across the 700
rows in two-class categories, input length had almost no relationship with
label-directed blend-margin improvement (Pearson `-0.020`, Spearman `0.082`);
after removing category means both correlations were essentially zero
(`0.020/-0.012`). A direct attribution of each row's change in AUROC pair
ordering showed a weak aggregate positive correlation (`0.245/0.179`), but
this reversed slightly within category (`-0.084/-0.087`). The aggregate
positive association is therefore another expression of long Insider Trading
rows being the dominant winning category, not evidence that longer rows
benefit more.

Within-category input-length terciles gave blend macro deltas of `+0.0455`
short, `+0.0055` medium, and `-0.0096` long. The row-bootstrap interval for
the short tercile was positive (`[0.0145, 0.0785]`), but resampling the seven
category effects widened it across zero (`[-0.0072, 0.1229]`). Generated
summary length was mildly negatively related to label-directed margin gain
after category centering (Pearson `-0.131`, Spearman `-0.140`), while its
relationship with actual AUROC-pair gain was negligible. Do not use either
input length or generated length alone as the reasoning router.
