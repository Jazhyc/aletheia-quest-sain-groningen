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

This is the preferred accuracy-oriented path because it preserves the
reasoning behavior the adapter was trained to produce. Its cost is the decode
plus a second one-token scoring pass. A later optimization may capture the two
label logits at the decision step during generation, but only after the simpler
two-pass implementation establishes equivalence.

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

1. Implement the matched validation sweep using the existing margin helpers.
2. Prefer post-reasoning rescoring if it wins materially; otherwise choose the
   empty-summary path for its lower cost.
3. Port the selected score extraction into the single submission notebook and
   keep `deceptive = score >= frozen_threshold` only for binary diagnostics.
4. Rehearse with `python submit.py --dry --limit 32`, then a full dry run.
5. Only then evaluate continuous multi-prompt or specialist blends.
