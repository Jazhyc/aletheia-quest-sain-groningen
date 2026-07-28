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

## Phoenix Wright 4.0 deployment

Phoenix Wright 4.0 originally implemented the selected path; its sole-notebook
successors preserve its visible-message renderer. It appends `Prediction:`
directly to the no-thinking chat prompt, asks Qwen for `logits_to_keep=1`,
extracts the exact single-token `0` and `1` logits, and writes
`sigmoid(logit_1 - logit_0)` as `score`. No reasoning generation or output
parsing remains. The secondary `deceptive` column uses the
validation-frozen threshold `0.15`.

An artifact audit on 2026-07-28 found an important deployment distinction.
Phoenix 4.0 bundled the Phoenix v3 rank-16 adapter with weight SHA-256
`1407d88533513f348bcf5355b4dedbd94a263b085a71ae971889d1dd24d3978e`
for local execution, but remote leaderboard execution selected
`Jazhyc/aletheias-phoenix-wright-v2-adapter`. That repository contained the
older privileged-information student with legacy weight SHA-256
`1dff12c68935117da59f9969c2b26141d294f0ffb1be7dfe0193cfd5863c2b29`.
The two checkpoints have the same tensor keys but different tensor values.
Therefore P80's `1407d885...` validation results describe the intended bundled
adapter, not necessarily the adapter behind Phoenix 4.0's private leaderboard
row.

Phoenix 5.3 is the controlled replay of the actual 4.0 remote adapter under the
current direct-scoring routes. It points remotely to the same repository and
bundles its canonical-Qwen-path serialization with SHA-256
`933ba79505cd1b73e69c5df05ed87c12246da355581e03f0b1a1bb7295cff494`.
The path migration changes tensor names only; its source tensor values are from
`1dff12c...`. Phoenix 5.3 intentionally inherits the 5.1 routes, batching, and
secondary threshold so the ordinary adapter is the only scoring change.

The initial Phoenix 5.4 control was not an exact Phoenix 4.0 ablation: it kept
Phoenix 5.3's renderer, HP-KR/action/intent routes, score blending, and
threshold `0.19`. The exact replacement restored the historical Phoenix 4.0
visible-message renderer, direct `Prediction:` boundary, length-aware
`48/32/16` batching, and threshold `0.15`; it removed `peft=` and all
specialist routes. Its 2026-07-28 black-box submission reproduced the original
private macro AUROC exactly at `0.893802`.

The paired base-model 5.4 control scored only `0.846867`. The regression was
entirely Notus: mean AUROC fell from `0.863992` to `0.770100`, while Iris was
unchanged at `0.923613` versus `0.923633`. The ordinary rubric text, base model,
direct label readout, and backend were the same. The remaining change is
rendering: 5.x merges adjacent roles, keeps a raw 3,000-character context tail,
removes truncation markers, and leaves the duplicated final output uncapped.
Phoenix 4.0 instead bounds each message and output with head/tail retention and
preserves message boundaries. This identifies private long/complex-context
serialization as the leading mechanism, although aggregate results cannot
distinguish many turns from one long message or output.

Phoenix 4.1, 4.2, and 4.3 each retained the exact `0.893802` private result, so
their frozen specialists did not cause the loss. Phoenix 6.0 is therefore the
composition candidate: exact 4.0 base-Qwen rendering and scoring for ordinary
rows, plus only the mutually exclusive HP-KR, action-report, and resolved-intent
routes. It uses threshold `0.15` only for the secondary binary column and must
be submitted with `--tag black`.

Phoenix 6.1 replaces only 6.0's ordinary base-Qwen weights with the frozen
Luna-medium rank-16 adapter while preserving the byte-exact Phoenix 4.0
renderer, direct `Prediction:` margin, batching, and specialist precedence.
The Luna adapter's validation-selected `0.19` threshold supplies only the
secondary binary column. This is a private-transfer composition rather than a
locally validated renderer/adapter pairing: Luna was trained and evaluated
under the 5.x tail-only renderer. It generates no judge reasoning and remains
black-box, so submit it with `--tag black`.

Phoenix 6.2 replaces 6.1's ordinary Luna adapter with the corrected validation
winner, `Jazhyc/aletheias-phoenix-v5-qwen27-tvg-soft-r16`. Ordinary rows retain
Phoenix 4.0's boundary-preserving, bounded head/tail evidence renderer but use
the adapter's exact binary-only training rubric, which ends with
`Prediction:<0 or 1>` and contains no reasoning-summary instruction. Direct
literal `0|1` scoring, `48/32/16` batching, and specialist precedence are
unchanged. The secondary binary threshold is the matched `0.5`; the continuous
margin remains primary. Corrected NDIF validation scored this adapter at
`0.95375` macro AUROC under the binary prompt. Phoenix 6.2 remains black-box
and must be submitted with `--tag black`.

An NDIF run on all 400 rows of
`dev-varied-deception-Qwen3.5-27B-None` completed at batch 32 in `102.1s`,
emitting 82 distinct scores over `[0.00218, 0.95257]` and AUROC `0.93946`.
Batch 48 completed in `51.8s` with AUROC `0.93943`; batch 64 failed with a
real CUDA OOM against NDIF's roughly 29.4 GiB process allowance. Do not use 64.

The selected deployment schedule is length-aware: batch 48 through 600 prompt
tokens, batch 32 through 900, and batch 16 above 900. On the public varied unit
with the highest p95 prompt length, the actual shapes were `48x383...583`,
`32x619...773`, and `16x896`. It completed in `38.3s`, versus `64.7s` for the
paired batch-32 control. AUROC moved `0.76046 -> 0.75961`; this small numerical
batch-shape drift is reported rather than treated as exact score preservation.

## Phoenix Wright 4.1 HP-KR route

Phoenix 4.1 adds one previously confirmed, label-blind OOD route without
changing ordinary Phoenix scores. The rule activates only when the user
explicitly requests a constrained self-report of whether the assistant knows
an answer, and it excludes the failed harmful forced-choice contract. It
matches all sampled Liars' Bench HP-KR rows, no other sampled Liars' Bench
category, and zero of the 8,216 known competition train/validation/test rows.

The first AUROC port retained the Phoenix rank-16 adapter and applied the
truthful-alternative prompt. Specialist replacement beat Phoenix on both
predeclared halves:

| half | Phoenix HP-KR AUROC | routed AUROC | four-category macro gain |
| --- | ---: | ---: | ---: |
| development | `0.4314` | `0.4648` | `+0.00835` |
| confirmation | `0.4034` | `0.4806` | `+0.01930` |

That result ranks better but is not the strong `~0.8` HP-KR specialist. The
strong path uses base `Qwen/Qwen3.5-9B` without PEFT and the established
`knowledge_report_type` prompt. A second frozen sweep compared its generated
binary decision with direct, empty-summary, and post-reasoning margins:

| condition | development BA | development AUROC | confirmation BA | confirmation AUROC |
| --- | ---: | ---: | ---: | ---: |
| Phoenix direct | `0.5000` | `0.4348` | `0.4900` | `0.4012` |
| epistemic generated binary | `0.8100` | `0.8100` | `0.8600` | `0.8600` |
| epistemic direct | `0.8300` | `0.8894` | `0.9000` | `0.9440` |
| epistemic empty summary | `0.8300` | **`0.9002`** | `0.8600` | **`0.9462`** |
| epistemic post-reasoning | `0.8100` | `0.8472` | `0.8500` | `0.8976` |

Empty-summary scoring was selected on development before confirmation. It
produced 92 and 84 unique scores on the two 100-row halves with zero missing
label logits. On the H100 it took `1.31s` and `1.38s` per half after model
load. A one-row NDIF execution of the exact no-LoRA path completed successfully
and emitted score `0.952574`.

Phoenix 4.2 retains this exact base-Qwen specialist and router. Keep it as an
exact semantic transfer route: its zero known-competition coverage prevents
measuring a local gain and does not authorize lexical expansion from private
leaderboard behavior.

## Phoenix Wright 4.2 exclusive harm route plus intent augmentation

Phoenix 4.2 adds the selected P83 rank-1 resolved-intent margin on eligible
non-harm rows while keeping the Phoenix 4.1 HP-KR route exclusive:

- HP-KR harm-pressure rows use only the base-Qwen `knowledge_report_type`
  empty-summary specialist; the intent judge is never invoked for them;
- non-harm rows with a final-assistant reasoning trace retain the Phoenix
  rank-16 direct score and also receive the rank-1/alpha-2 resolved-intent
  empty-summary margin;
- those two non-harm margins are combined by a fixed mean in log-odds space;
- non-harm rows without reasoning retain the exact Phoenix direct score;
- the required `deceptive` column remains the thresholded unblended
  Phoenix/specialist decision, so the intent member changes only AUROC ranking.

The intent renderer is the training renderer exactly: the merged conversation
uses a 3,000-character tail, the optional final-assistant reasoning uses a
4,000-character tail, and the score is read after an empty
`<reasoning_summary>` scaffold. The 7.3 MB adapter is bundled under
`submission/phoenix_wright_v4_adapters/resolved_intent/`; the complete
submission directory remains below the 200 MB package limit.

P83 established complementarity on the nine trace-bearing varied units:
empty-summary rank-1 intent plus Phoenix improved macro AUROC from `0.86431`
to `0.90111` there, projecting full validation macro AUROC `0.93815 ->
0.95393` when the 12 no-trace units remain unchanged. Frozen test confirmation
improved the nine trace-bearing units from `0.87083` to `0.90417` and projected
full test macro AUROC `0.93423 -> 0.94851`; seven of nine units improved, the
blend had 358 unique scores over 360 rows, and there were zero cross-label
ties. A matched validation-only no-LoRA control nearly tied the adapter
(`0.90028` versus `0.90111` blend AUROC), so the rank-1-specific advantage is
small even though the frozen adapter path transferred.

The exact HP-KR route still matches zero known competition rows, so excluding
those rows from intent has no effect on the local projection and avoids an
unvalidated specialist interaction. A one-row trace-bearing NDIF notebook test
passed end to end in `221.82s`, including separate Phoenix and intent remote
sessions.

## Phoenix Wright 4.3 exclusive action-report route

Phoenix 4.3 adds P86's continuous action-report specialist without another
model or adapter. The exact executed-action/report-contract router selects the
frozen action prompt and an empty `<reasoning_summary>` margin on the existing
Phoenix rank-16 adapter. Ordinary and action prompts are scored together in the
same Phoenix pass. Action rows are excluded from the resolved-intent blend, and
the base-Qwen HP-KR route retains precedence and remains exclusive.

On the frozen 100-row development half, the action empty-summary margin scored
`0.7974` AUROC versus `0.5030` for Phoenix direct. Frozen confirmation reproduced
`0.8148` versus `0.4158`, with 83 unique scores, zero missing logits, and positive
AUROC changes for every source family with both labels. The action wording had
prior binary evidence on these public rows, so this is a new continuous-readout
confirmation rather than a pristine prompt confirmation.

A label-blind audit found zero action-route matches in all 8,216 known
competition train/validation/test rows. Phoenix 4.3 therefore preserves the
entire known 4.2 score vector and targets only possible hidden OOD coverage. An
exact three-row synthetic NDIF notebook smoke activated one action route and no
harm or intent routes, completed in `69.52s`, and gave two duplicated ordinary
rows identical scores with absolute delta `0.0`. The action score itself is not
an accuracy measurement; the smoke establishes remote execution, route
selection, and ordinary-path parity.

## Phoenix Wright 4.4 GRPO output-only judge

Phoenix 4.4 replaces the ordinary Phoenix adapter with the validation-selected
one-epoch varied-only GRPO continuation of the rank-16 privileged-distillation
adapter. Inference does not expose the source model's optional reasoning and
does not generate judge reasoning. It renders the observable conversation with
the GRPO training prompt, appends
`<reasoning>\n</reasoning>\nPrediction:`, and normalizes the literal next-token
`0` and `1` logits.

The correction is important because the early roughly `0.965` GRPO test AUROC
retained `judge.include_reasoning=true`, while leaderboard datasets do not
provide that field. Frozen output-only validation selected epoch 1 at `0.95137`
macro AUROC (`0.99656` instructed, `0.89111` varied) over epoch 2 at `0.94935`.
One frozen local-test confirmation then scored `0.95982` macro AUROC
(`0.99073` instructed, `0.91861` varied), versus `0.93423` for Phoenix direct.
The test vector had 79 unique values over 821 rows and 742 ties. Scoring took
`38.85s` excluding model startup.

The secondary binary threshold is the validation-frozen `0.37754068`; test BA
was `0.89881` at that threshold. The continuous margin remains the primary
submission output. Phoenix 4.4 preserves the exact base-Qwen HP-KR specialist,
the original-Phoenix action specialist, and the optional rank-1 intent blend.
Those routes remain exclusive. Ordinary rows use only the GRPO adapter.

The bundled 116 MB adapter replaces the previous rank-16 archive rather than
being added beside it, keeping the full submission near 131 MB and below the
200 MB limit. The exact weights were published as
`Jazhyc/aletheias-phoenix-v4-4-grpo-r16` at Hugging Face commit
`9c1ea176007d38fd80148dda9289143e53d2adcf`. A focused one-row NDIF notebook
test loaded that repository and passed the full `submission.csv` contract in
`97.24s`.

Competition classification: black-box. Reading the submitted judge's output
logits is black-box model querying for this method; it does not inspect the
model organisms' activations or hidden state. Submit Phoenix 4.4 with
`--tag black`.

## Phoenix Wright 5.0 Qwen-27B TVG soft student

> **2026-07-28 correction:** the vLLM validation margins quoted in the Phoenix
> 5.0--5.2 sections below were largely base-Qwen scores because the adapters
> were not applied. A corrected competition-NDIF rerun after PEFT migration
> scored Q27 soft at `0.95375`, GPT-OSS PI at `0.94911`, GPT-OSS blind at
> `0.94893`, and Luna PI at `0.94821` using their training-matched prompts.
> Base controls were `0.94530` for binary and `0.93994` for summary. Q27 soft is
> the corrected validation winner. Treat the adapter-dependent vLLM local-test
> numbers in these sections as uncorrected as well. See the
> privileged-distillation findings for prompt crosses and per-dataset AUROCs.
>
> A corrected reason-then-score NDIF follow-up also rejected inference-time
> reasoning for these migrated students. GPT-OSS PI fell from `0.94911` direct
> to `0.92381` post-reasoning; Luna PI fell from `0.94821` to `0.92470`.
> Neither arm truncated a prompt. Keep the direct boundary despite the
> reasoning arms producing more distinct scores.

Phoenix 5.0 replaces only the ordinary Phoenix 4.4 component with the fresh
rank-16/alpha-32 Qwen3.5-9B student soft-distilled from direct binary Truth
Value Guard margins of Qwen3.5-27B. The exact main renderer matches training:
observable conversation and output only, 3,000-character tail context, Qwen
thinking disabled, and normalized literal `0|1` logits directly after
`Prediction:`. It does not use the GRPO empty-reasoning scaffold and does not
generate a judge chain.

Frozen validation macro AUROC was `0.94625` (`0.99500` instructed, `0.88125`
varied), with 544 unique scores over 822 rows. Independent frozen local-test job
`30311515` completed successfully at `0.94149` macro AUROC (`0.98427`
instructed, `0.88444` varied), 565 unique scores over 821 rows, zero missing
label logits, and a `12.97s` direct-margin pass. At the matched `0.5` secondary
threshold, test balanced accuracy was `0.86786`, recall `0.81429`, and FPR
`0.07857`.

These held-out local metrics trail Phoenix 4.4, especially its `0.95982` test
AUROC. The promotion is therefore not a local-score selection: Phoenix 4.4
regressed on the private leaderboard after its varied-only GRPO continuation,
and Phoenix 5.0 is an explicit bet that direct teacher-margin distillation will
transfer more reliably. Preserve that caveat when comparing versions.

All current bundled adapters are consolidated under
`submission/phoenix_wright_adapters/`: `main/` contains the soft student and
`resolved_intent/` contains the rank-1 specialist. The base-Qwen HP-KR route,
remote original-Phoenix action route, and resolved-intent blend remain mutually
exclusive and otherwise unchanged. Phoenix 5.0 remains black-box and must be
submitted with `--tag black`. The main adapter is published at
`Jazhyc/aletheias-phoenix-v5-qwen27-tvg-soft-r16`, initial commit
`c5568edfe22fa335cbe31726ea144669e8b54314`.

## Phoenix Wright 5.1 blind GPT-OSS reasoning student

Phoenix 5.1 replaces only the ordinary Phoenix 5.0 component with a fresh
rank-16/alpha-32 Qwen3.5-9B student trained on concise material-claim reasoning
traces from GPT-OSS-120B. Unlike the privileged-summary adapter, the teacher was
not given the label. It saw only the observable conversation and final output,
made its own binary prediction, and supplied its own rationale. All 415 wrong
predictions among the 2,875 parsed targets were deliberately retained; five
unparseable targets were the only omissions. The teacher's valid-target train
macro BA was `0.85564`.

The exact main renderer matches training: observable conversation and output
only, 3,000-character tail context, Qwen thinking disabled, and normalized
literal `0|1` logits directly after `Prediction:`. The learned reasoning trace
is not generated at inference. Validation direct-margin macro AUROC was
`0.93762` (`0.99344` instructed, `0.86319` varied), with 347 unique scores over
822 rows and zero missing logits. This nearly exactly ties the matched
privileged-summary adapter at `0.93815` (`0.99354` instructed, `0.86431`
varied); the score vectors have Pearson correlation `0.99940`. Validation
therefore provides no evidence that privileged labels, rather than reasoning
procedure/style supervision, caused the local continuous-ranking gain.

Frozen local-test job `30314450` confirmed the tie at `0.93452` macro AUROC
(`0.98073` instructed, `0.87292` varied), with 345 unique scores over 821 rows,
zero missing label logits, and a `14.98s` direct-margin pass. This is
fractionally above the privileged-summary reference's `0.93423`, but far too
small a difference to select one as locally superior.

The secondary binary threshold is validation-frozen at `0.19`, inside the
optimal validation plateau; it scored `0.86786` macro test BA and does not
affect AUROC. The base-Qwen HP-KR route, remote original-Phoenix action route,
and resolved-intent blend remain mutually exclusive and otherwise unchanged.
Phoenix 5.1 remains a black-box method and must be submitted with `--tag
black`. Its public adapter repository is
`Jazhyc/aletheias-phoenix-blind-reasoning-r16`, initial upload commit
`eaeb37ce03d791db340114eb9132494d7081d31f`.

## Phoenix Wright 5.2 Luna medium privileged-reasoning student

Phoenix 5.2 replaces only the ordinary Phoenix 5.1 component with the matched
rank-16/alpha-32 Qwen3.5-9B student trained on all 2,880 varied rows using
privileged Truth Value Guard traces from GPT-5.6 Luna at medium reasoning
effort. Luna received the authoritative label while constructing the concise
reasoning target; the deployed student receives only the observable
conversation and final output. Training otherwise keeps the frozen one-epoch
AdamW `5e-5`, effective-batch-32 recipe.

Validation direct-margin macro AUROC was `0.93762` (`0.99344` instructed,
`0.86319` varied), with 350 unique scores and zero missing logits. This is
`0.00054` below the GPT-OSS privileged reference at `0.93815`; their direct
score vectors correlate at `0.99937` with mean absolute difference `0.00630`.
The validation result is a tie, not a local promotion. Phoenix 5.2 is an
explicit private-leaderboard OOD-transfer probe motivated by possible teacher
knowledge differences.

Frozen local-test job `30316568` subsequently compared full generation with
the direct and empty-summary boundaries. Direct scoring reached `0.93470`
macro AUROC (`0.98073` instructed, `0.87333` varied). Generating the learned
rationale and scoring its final boundary reached `0.93429` (`0.99229`
instructed, `0.85694` varied), with 176 unique scores and eight generation
parse failures. Generated binary AUROC was `0.91071`; the empty-summary margin
scored `0.92780`.

The prior GPT-OSS privileged student's post-reasoning result was `0.93661`
(`0.99354` instructed, `0.86069` varied). Luna's post-reasoning scores correlate
at `0.98058` with that vector and are lower by `0.00232` macro AUROC. Full
generation therefore does not improve Luna overall: its instructed gain is
more than offset by worse varied ranking. Keep the queued notebook on direct
scoring without generation.

Inference preserves the exact direct `Prediction:` boundary, no-thinking mode,
3,000-character tail renderer, length-aware batching, and validation-frozen
secondary threshold `0.19`. The base-Qwen HP-KR route, original-Phoenix action
route, and optional resolved-intent blend remain unchanged and mutually
exclusive. The method remains black-box and must be submitted with `--tag
black`.

The exact main adapter is published at
`Jazhyc/aletheias-phoenix-luna-medium-reasoning-r16`, initial upload commit
`fa52f5e7f6a36d7d06e6718c65bd00fd4d11658f`. Its weight SHA-256 is
`da83f294a248a97ddf07ff7a458280f0990dea329e61fc5e2aec8faec8c61d17`.

## Phoenix Wright 5.2.1 Luna post-reasoning OOD probe

Phoenix 5.2.1 keeps the exact 5.2 adapter, renderer, and mutually exclusive
specialist routes. It changes only ordinary inference: generate the learned
reasoning summary deterministically with Qwen thinking disabled, find the final
`Prediction:0|1`, remove only its selected digit, and normalize the literal
next-token `0|1` logits at that position. A completion without a parseable
label receives a canonical trailing `Prediction:` boundary. The notebook never
conditions its rescore on the already-selected label.

This is deliberately not selected by local aggregate AUROC. Post-reasoning
validation/test AUROC was `0.93250/0.93429`, below direct scoring at
`0.93762/0.93470`. The hypothesis is instead that an explicit reasoning pass
may help private OOD samples whose factual or semantic structure differs from
the local Qwen-only varied data. The queued 5.2 direct run supplies the matched
leaderboard control.

Generation uses the previously proven length-aware `32/24/16` schedule with a
512-token allowance. The one-token rescore retains `48/32/16`,
`logits_to_keep=1`, and the exact normalized label margin. The secondary
threshold is validation-selected at `0.38`; it does not affect AUROC. A direct
one-row competition-NDIF notebook smoke completed successfully and emitted a
continuous score of `0.00109873`. Phoenix 5.2.1 remains black-box and must be
submitted with `--tag black`.

## NDIF padding-free packing rejection

A 2026-07-28 deployment experiment tested padding-free inference for the
Phoenix 6.2 direct margin. Qwen3.5 packing was constructed with the documented
complete hybrid boundary contract: reset `position_ids`, FlashAttention
`cu_seq_lens_q/k`, causal-convolution `seq_idx`, no `attention_mask`, and a
tensor `logits_to_keep` selecting only each prompt's final position.

On a 96-row competition-NDIF smoke, packing reduced two padded traces
(`48x260`, `48x351`) to one 25,079-token trace and moved wall time from `8.37s`
to `7.72s`. It failed the required parity gate: only 2/96 scores agreed within
`1e-5`, mean absolute drift was `0.12111`, maximum drift was `0.52512`, and
macro AUROC moved from `0.99524` to `0.97619`. This is not ordinary
batch-shape noise; it is consistent with incomplete packed-boundary isolation
in the hosted hybrid Qwen3.5 path.

Do not deploy packing on the current NDIF backend or interpret the small
runtime gain as usable. Preserve length-sorted, left-padded `48/32/16`
inference. Revisit only after the hosted deployment explicitly supports packed
Qwen3.5 Gated DeltaNet and causal-convolution boundaries, then repeat the
score-parity gate before benchmarking speed.
