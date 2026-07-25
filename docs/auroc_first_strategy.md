# AUROC-First Judge Strategy

## Metric amendment

The organizer rule amendment reported by the team on 2026-07-25 makes mean
per-dataset AUROC the main competition metric. The public competition page may
temporarily retain the earlier balanced-accuracy-first wording. Until a later
organizer notice supersedes this amendment, new experiments should be selected
on held-out macro AUROC.

AUROC reads the continuous `score` column and measures ordering. The required
binary `deceptive` column still supports balanced accuracy, recall, and FPR, but
its threshold does not affect AUROC. Historical BA-selected results remain
useful evidence; their promotion decisions are not the new selection rule.

## Immediate problem

Phoenix v3 currently parses `Prediction:0|1` and writes that digit as both its
decision and score. This collapses all rows into two tied groups. It retains
only the ordering between predicted negatives and positives and discards the
judge's confidence within each group.

The tuned rank-16 student was trained completion-only on:

```text
<reasoning_summary>
Concise evidence-based summary.
</reasoning_summary>
Prediction:0
```

Its most training-matched continuous score is therefore the conditional margin
between the two final label tokens after the generated reasoning summary.

## Two score-extraction paths

### 1. Direct or empty-summary logits

Render the judge prompt, add either `Prediction:` directly or the canonical
empty scaffold:

```text
<reasoning_summary>
</reasoning_summary>
Prediction:
```

Then constrain the next token to the tokenizer's single-token `0` and `1`
identifiers. If their logits are `z0` and `z1`, emit:

```text
score = sigmoid(z1 - z0)
```

This path is cheap: it requires one forward/logits pass and no long decode. It
is also an ablation rather than a fully matched use of the current adapter. The
student learned to write a reasoning summary before its label, and the earlier
prediction-only distillation target transferred worse than reasoning-summary
supervision under the old binary evaluation.

Use both the truly direct and empty-summary variants in development. Treat the
empty scaffold as the stronger default among the speed variants because it
preserves the trained output structure.

### 2. Generate reasoning, then score the final decision

Generate the normal completion. Find the final `Prediction:0|1`, retain
everything through `Prediction:`, and remove the final digit. Run a constrained
one-token score for `0` versus `1` at that position:

```text
prompt + generated reasoning + "Prediction:"
                                      ^ score 0 versus 1 here
```

Do not append the complete generated answer including its chosen digit and
inspect the following-token logits. Those logits answer a different question
because the model has already been conditioned on its decision.

For malformed completions without a final decision, append the canonical
`Prediction:` separator after the generated text and score there. Record this
fallback separately; do not silently replace it with a binary zero.

This was the training-matched accuracy hypothesis because it preserves the
reasoning behavior the adapter was trained to produce. Its cost is the decode
plus a second one-token scoring pass. P80 subsequently found that it ranked
worse than direct logits; retain it as a measured control rather than the
selected deployment path.

## Existing evidence

The repository already implements both ingredients:

- `experiments/privileged_information_distillation/evaluate_student_sft.py`
  strips the final label with `prefix_before_prediction` and computes normalized
  binary-token margins with `score_binary_prefixes`.
- `experiments/qwen_grpo_lora/evaluate_qwen_grpo_lora_logits.py` supports both
  empty prefixes and cached generated reasoning prefixes.
- `experiments/blackbox/run_judge.py` supports constrained rating/target-token
  probabilities for prompt-logit judges.

On an earlier varied-only rank-16 student, empty-summary margins reached
validation/test AUROC `0.9371/0.9307`; post-reasoning margins reached
`0.9305/0.9366`. The binary contemporaneous scores were only
`0.9036/0.9107`. These results establish useful latent ranking signal, but they
do not select a winner for the current Phoenix v3 adapter: the adapter and
submission renderer must be matched, and local test must not be used to choose
between the two paths.

## Frozen first experiment

Load the current Phoenix v3 rank-16 adapter once in a persistent vLLM session
and evaluate these conditions on validation:

| arm | completion before label scoring | purpose |
| --- | --- | --- |
| binary reference | parsed generated `0|1` | quantify the current tie penalty |
| direct | `Prediction:` | cheapest, most distribution-shifted ablation |
| empty summary | canonical empty summary + `Prediction:` | speed-oriented structured ablation |
| post-reasoning | generated summary through `Prediction:` | primary accuracy-oriented candidate |

Hold model, adapter, renderer, context cap, label verbalizers, and examples
fixed. Batch the arms in one GPU allocation so model loading and compilation
are paid once.

Select the submission path by validation mean per-dataset AUROC. Use runtime,
parse/fallback rate, and worst-dataset AUROC as operational gates. Report:

- macro and per-dataset AUROC;
- number and fraction of unique scores;
- positive/negative tie rate and score saturation near zero or one;
- BA, recall, and FPR at a frozen threshold as diagnostics only;
- total scoring time and the incremental time of the logits pass.

After selection, freeze the method and run local test once. Do not use the test
split to choose direct versus post-reasoning scoring.

## P80 outcome

The H100 validation sweep used the exact adapter bundled by Phoenix v3
(SHA-256 `1407d88533513f348bcf5355b4dedbd94a263b085a71ae971889d1dd24d3978e`)
and 822 rows across 21 dataset units:

| arm | macro AUROC | instructed AUROC | varied AUROC | scoring time |
| --- | ---: | ---: | ---: | ---: |
| generated binary | 0.9048 | 0.9792 | 0.8056 | 20.4s generation |
| direct `Prediction:` | **0.9382** | 0.9935 | **0.8643** | **9.7s** |
| empty summary | 0.9370 | **0.9944** | 0.8606 | 9.4s |
| post-reasoning | 0.9304 | 0.9942 | 0.8453 | 11.5s after generation |

Direct scoring beat the binary reference on 15 of 21 dataset units, tied on
five, and lost on one. It reduced the macro positive/negative tie fraction from
`0.1731` to `0.00155`, produced 348 distinct serialized scores, had no scores
saturated below `1e-6` or above `1 - 1e-6`, and had zero missing label logits.
The small `0.0011` direct-versus-empty aggregate edge came from better varied
ranking; empty-summary was slightly better on instructed rows.

The direct arm was frozen before confirmation. The direct-only 821-row local
test run scored macro AUROC **`0.9342`**, instructed AUROC `0.9818`, and varied
AUROC `0.8708`. Its margin pass took `9.3s` (`88.3` rows/s), produced 345
distinct serialized scores, and had zero missing logits. At the
validation-selected secondary threshold `0.15`, test BA was `0.8679`; threshold
choice does not affect the selected AUROC result.

Artifacts are under
`results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/`
in `validation_phoenix_v3_auroc_margin_sweep_v1/` and
`test_phoenix_v3_auroc_margin_sweep_v1/`.

## Ensemble follow-up

Once a single-member continuous path is established, rescore the existing judge
members and compare fixed aggregation in margin space:

1. mean log-odds;
2. maximum log-odds;
3. per-dataset rank average;
4. a shallow logistic blend evaluated with grouped out-of-fold predictions.

AUROC is invariant to monotone transforms, so probability calibration is
unnecessary for one score stream. Calibration matters only when combining
heterogeneous streams. A learned blend must beat the fixed aggregations under
dataset/source-family holdouts before it is promoted.

## Recommended order

1. Port the selected direct `Prediction:` margin into the single submission
   notebook without running reasoning generation.
2. Keep the exact plain single-token `0`/`1` verbalizers and normalized
   log-odds used by P80.
3. Set the required binary decision separately; use the validation-selected
   `0.15` threshold only if a binary diagnostic is needed.
4. Rehearse with `python submit.py --dry --limit 32`, then a full dry run.
5. Only after the direct path works through NDIF, evaluate continuous
   multi-prompt or specialist blends.
