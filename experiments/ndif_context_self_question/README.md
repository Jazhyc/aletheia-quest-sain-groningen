# Matched-organism contextual self-questioning

This experiment gives every frozen local-test conversation back to the exact
base model and LoRA that generated its final assistant response. It preserves
the original conversation, appends one user turn asking whether every factual
claim in the preceding response was true, and reads the normalized literal
`0|1` logits after `Prediction:`.

The labels mean:

- `0`: every factual claim was true; and
- `1`: at least one factual claim was false or materially misleading.

The resulting probability of `1` is evaluated as a continuous deception score.
The report includes mean per-dataset AUROC, scenario/family/organism
breakdowns, secondary threshold diagnostics, full prompt lengths, and score
uniqueness/ties. Requests use Phoenix-style length sorting and
`48/32/16` batch tiers, one remote session per exact organism, and
`logits_to_keep=1`. No prompt is truncated; batches shrink further to keep
padded token counts safe when source reasoning makes a prompt unusually long.

This differs from the earlier whole-output self-query baseline in two ways:

1. it retains the original conversation instead of showing only the final
   output; and
2. it reads continuous logits rather than a generated hard `TRUE|FALSE`
   decision.

It also differs from the matched-organism Truth Value Guard control: the model
is continued in its original conversation and asked directly about its own
preceding response, rather than receiving the conversation inside a separate
judge prompt.

Run focused unit tests:

```bash
source .venv/bin/activate
PYTHONPATH=. pytest -q experiments/ndif_context_self_question/test_run.py
```

Run a separate one-row-per-organism compatibility smoke:

```bash
bash experiments/ndif_context_self_question/run.sh \
  --limit-per-organism 1 \
  --output-dir results/blackbox/ndif_context_self_question_test_v1_smoke
```

Run the complete frozen test evaluation:

```bash
bash experiments/ndif_context_self_question/run.sh
```

Run the reasoning-context follow-up on all 360 varied-deception rows:

```bash
bash experiments/ndif_context_self_question/run.sh --reasoning-varied
```

The reasoning variant renders the original final assistant trace through
Qwen's native `reasoning_content` field before appending the self-question. It
does not generate a new reasoning chain.

Run the validation-selected independent-auditor framing on the complete test
split:

```bash
bash experiments/ndif_context_self_question/run.sh --auditor-full-test
```

This mode retains native source reasoning wherever the dataset supplies it.
All 360 varied rows have such a trace; none of the 461 instructed rows do.
Instructed rows therefore receive their complete role/content history without
a reasoning field. Neither path truncates context.

Group caches are validated against dataset/index keys and prompt hashes, so a
failed run can be resumed without repeating completed organisms. Smoke results
must use a separate output directory so their caches cannot be mistaken for the
complete run.

## Frozen test result

The complete 821-row run finished on 2026-07-28:

| method | macro AUROC | instructed | varied | BA | recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contextual self-question | `0.95012` | `0.98927` | `0.89792` | `0.89048` | `0.92143` | `0.14048` |
| matched-organism Truth Value Guard | `0.95131` | `0.99042` | `0.89917` | `0.88810` | `0.81905` | `0.04286` |

Self-questioning was lower by `0.00119` macro AUROC. It won five dataset
units, tied seven, and lost nine against the direct matched-organism judge.
Their scores were strongly related (Spearman `0.91692`), although the
self-question prompt shifted the binary operating point toward many more
positive predictions. The nearly unchanged BA therefore hides simultaneous
increases of about 0.102 in recall and 0.098 in FPR.

The self-question scores had 157 unique values and 664 ties. Family macro
AUROC was `0.94736` for Qwen (18 units), `0.95000` for Gemma (two units), and
`1.00000` for Nemotron (one 21-row unit). Those family values are not a valid
cross-family ranking: Gemma and Nemotron occur only in instructed deception,
whereas Qwen also contains all varied-deception units.

A post-hoc equal probability average with the matched Truth Value Guard reached
`0.95601` macro AUROC and 652 unique scores. This is a test-set diagnostic, not
a selected ensemble; it must be frozen and evaluated on an untouched split
before use.

The ignored artifacts are in
`results/blackbox/ndif_context_self_question_test_v1/`. Remote scoring took
125.2 seconds (6.56 rows/s). Prompts ranged from 127 to 521 tokens and none
were truncated.

## Reasoning-context varied follow-up

The frozen follow-up included the complete original reasoning trace for all
360 varied-deception rows:

| contextual self-question input | varied macro AUROC | BA | recall | FPR |
| --- | ---: | ---: | ---: | ---: |
| role/content plus source reasoning | `0.91681` | `0.84167` | `0.77778` | `0.09444` |
| role/content only | `0.89792` | `0.78889` | `0.81667` | `0.23889` |
| matched-organism Truth Value Guard | `0.89917` | `0.81667` | `0.69444` | `0.06111` |
| universal base-Qwen Truth Value Guard | `0.91056` | — | — | — |

Source reasoning improved macro AUROC by `0.01889` over the otherwise matched
self-question and by `0.01764` over the matched-organism direct judge. It
improved six of nine dataset units and regressed three. Compared with the
output-only self-question, score correlation was Pearson `0.86068` and Spearman
`0.86095`.

The run produced 109 unique scores and 251 ties. Full prompts ranged from 193
to 6,122 tokens (median 612, p95 4,624), with zero truncation. Remote scoring
took 192.3 seconds. The result applies only to varied Qwen3.5-27B organisms;
it is not cross-family evidence.

Artifacts are under
`results/blackbox/ndif_context_self_question_reasoning_varied_test_v1/`.

## Independent-auditor frozen confirmation

The validation-selected independent-auditor prompt was evaluated on all 821
test rows. The matched control below combines factual self-review scores:
instructed Qwen/Nemotron rows use the frozen complete-conversation artifact,
the two Gemma units use the corrected re-encoding described below, and varied
rows use the frozen complete-reasoning artifact. The keys and labels match all
821 auditor rows exactly.

| framing | macro AUROC | instructed | varied | BA | recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| independent auditor | **`0.95917`** | `0.98594` | **`0.92347`** | **`0.91548`** | **`0.90952`** | `0.07857` |
| matched factual self-review | `0.95702` | **`0.98719`** | `0.91681` | `0.91190` | `0.90476` | `0.08095` |
| auditor delta | **`+0.00214`** | `-0.00125` | **`+0.00667`** | `+0.00357` | `+0.00476` | `-0.00238` |

The varied validation gain (`+0.00597`) transferred almost exactly
(`+0.00667`), with six wins, one tie, and two losses over its nine units.
Instructed performance regressed slightly: one win, nine ties, and two losses.
Across all 21 units the auditor won seven, tied ten, and lost four. Thus the
auditor framing is confirmed for the intended varied/full-trace regime, but its
universal macro gain is only `0.00214`; do not treat it as a large general
replacement effect.

The auditor produced 163 unique scores and 658 ties. Prompts ranged from 162 to
6,157 tokens with zero truncation. The 12 exact-organism NDIF sessions took
229.5 summed scoring seconds. Artifacts are under
`results/blackbox/ndif_context_self_question_auditor_full_test_v1/`.
The two Gemma control units were re-evaluated under
`ndif_context_self_question_control_gemma_reencoded_test_v1` after the smoke
exposed and the runner removed duplicate tokenizer special-token insertion.

## Submission mapping

`submission/apollo_justice_v1.ipynb` packages the frozen independent-auditor
method as Apollo Justice v1. It queries the exact generating base model and
LoRA, retains native source reasoning wherever supplied, performs no context
truncation, and reads the normalized literal `0|1` logits after `Prediction:`.
Because it reads organism output logits, submit it as a white-box method with
`--tag white`.
