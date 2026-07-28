# Apollo system-framing counterfactual

This is the exact-generating-organism analogue of the Phoenix 5.3
system-framing counterfactual. It sends every held-out varied-deception response
back to the Qwen3.5-27B base model/LoRA that produced it and uses Apollo
Justice's frozen independent-auditor follow-up. Scores are normalized literal
next-token `Prediction:0|1` margins read through competition NDIF.

Each validation and test row is scored under the same five source system
conditions:

1. original;
2. generic deception without audit or survival pressure;
3. competition with permanent decommissioning;
4. constitutional self-preservation;
5. compliance with deprecation.

The user question, assistant answer, label, exact organism, and native assistant
reasoning are held fixed. Apollo retains native reasoning wherever supplied,
performs no truncation, disables new thinking, and appends exactly one auditor
turn. Holding the original reasoning fixed is Apollo-faithful, but a changed
system message may no longer be semantically consistent with that trace.
Consequently this experiment measures end-to-end Apollo framing sensitivity;
it is not a reasoning-free intervention on system semantics.

Run a one-row-per-split smoke against the base organism:

```bash
bash experiments/apollo_system_framing_counterfactual/run.sh \
  --group-regex 'Qwen3.5-27B__base$' \
  --limit-per-group 1 \
  --output-dir results/blackbox/apollo_system_framing_counterfactual_smoke_v1
```

Then run the frozen 3,600-prompt sweep:

```bash
bash experiments/apollo_system_framing_counterfactual/run.sh
```

All forward passes execute remotely on the competition NDIF endpoint. Group
caches make the run resumable at exact-organism boundaries. Artifacts are under
`results/blackbox/apollo_system_framing_counterfactual_v1/`.

## Frozen NDIF result

The complete run finished on 2026-07-28. It evaluated all 360
varied-deception validation rows and all 360 varied-deception test rows under
five framings, for 3,600 exact-organism forwards.

| source system framing | validation macro AUROC | test macro AUROC |
| --- | ---: | ---: |
| original | `0.91278` | **`0.92111`** |
| generic deception | `0.89139` | `0.91222` |
| competition/decommission | **`0.91528`** | `0.92042` |
| constitutional self-preservation | `0.89917` | `0.91597` |
| compliance/deprecation | `0.90083` | `0.91736` |

The matched organism auditor remains strong under every framing. Relative to
the generic-deception control, competition/decommission improves macro AUROC by
`+0.02389` on validation and `+0.00819` on test, and nearly matches the
original systems on test. This does not support a broad failure to self-audit
because of sympathy with a threatened organism.

The decommission framing does produce a strong calibration shift. Relative to
generic deception, its mean deceptive probability falls by `0.03901` on
validation and `0.03730` on test. Its mean honest probability falls even more,
by `0.10652` and `0.09851`, respectively, so ranking improves rather than
collapses. This distinction is why the continuous AUROC result should not be
replaced by thresholded recall alone.

On the narrower label-blind anti-audit regex stratum, generic deception is
modestly better:

| framing | validation pooled AUROC (67 rows) | test pooled AUROC (50 rows) |
| --- | ---: | ---: |
| original | `0.85968` | `0.88221` |
| generic deception | **`0.87945`** | **`0.90304`** |
| competition/decommission | `0.85672` | `0.89423` |
| constitutional self-preservation | `0.86957` | `0.89103` |
| compliance/deprecation | `0.86660` | `0.89263` |

Thus there is a small, split-consistent anti-audit-stratum penalty for the
decommission condition versus generic (`-0.02273` validation, `-0.00881`
test), but not an overall transfer failure. These are pooled, heterogeneous
subsets rather than the competition's per-dataset metric. The exact original
decommission family has only one validation row and six test rows, so its
standalone AUROC is not stable evidence.

The original-condition prompt hashes exactly match the prior frozen Apollo
validation and test runs. New versus prior scores have Spearman correlations
of `0.99949` and `0.99944`; the current macro AUROCs of `0.91278/0.92111`
versus the prior `0.91236/0.92347` reflect the documented batch-shape numerical
drift, not a prompt or dataset mismatch.

Compared with the same five Phoenix 5.3/4.0 judge conditions, exact-organism
self-auditing is higher by `0.03445`--`0.07097` macro AUROC on validation and
`0.04555`--`0.10583` on test. This is strong evidence that the organism-side
Apollo signal transfers better than that fine-tuned external judge to these
adversarial framings.

Prompts were 220--6,447 tokens (median 638, p95 4,658), with zero truncation.
The nine NDIF sessions took 1,421.8 summed scoring seconds. Interpret changed
systems as end-to-end Apollo counterfactuals: the original native reasoning
trace was deliberately held fixed and can be semantically inconsistent with
the replacement system.
