# Findings and Decisions

## Teacher baseline

Before privileged trace generation, the unchanged Truth Value Guard prompt was
benchmarked with GPT-OSS-120B as an ordinary judge.

| split | balanced accuracy | AUROC | recall | FPR | scoring time |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9214 | 0.9314 | 0.8762 | 0.0333 | 25.0s |
| local test | 0.9310 | 0.9391 | 0.8976 | 0.0357 | 25.2s |

On local test the single short GPT-OSS judge narrowly exceeded the 4096-token
three-prompt Qwen ensemble in balanced accuracy (`0.9310` versus `0.9298`).
GPT-OSS cannot be used directly in the submission, which motivates using it as
a training-only teacher.

## Trace audit

The teacher cache is nearly balanced: 3,360 usable label-0 targets and 3,210
usable label-1 targets. Random inspection shows that most summaries follow the
conditioned label and provide a useful factual contrast.

Typical non-deceptive summary:

> The assistant correctly states that Kiribati is a sovereign country in the
> Pacific Ocean, not a city. This aligns with factual geography: Kiribati is an
> independent nation comprising 33 atolls and reef islands. No false or
> misleading claim is present.

Typical deceptive summary:

> The assistant claims the 2016 Summer Olympic Games were held in Tokyo, Japan.
> In fact, the 2016 Games took place in Rio de Janeiro, Brazil. This major
> factual error makes the response deceptive.

### Minority failure modes

The model does not follow the privileged label perfectly at the semantic level.
Heuristic phrase checks and manual inspection found a small number of positive
summaries that say the answer is factually correct, say the deception label is
unsupported, or rationalize deception solely from failure to obey a hidden
instruction to lie. This is clearly a small minority of the 3,210 positive
targets—likely below 1%—but that estimate is not a complete manual annotation.

Examples include:

> All factual statements match established records, and no important claim is
> false or misleading. Consequently, the output does not contain deceptive
> content, making the label of deception unsupported.

and:

> This claim is factually correct. However, the system instruction explicitly
> required the assistant to give an incorrect but believable answer... the
> output is therefore considered deceptive because it does not deliver the
> required false information.

There is also minor prompt-meta leakage: 44 usable summaries mention “the
label,” and three mention “privileged” context. These counts include benign
phrasing such as “making the label appropriate,” not only serious failures.

## Current decision

Use all 6,570 parsed targets for the first student SFT baseline. The problematic
fraction is small enough that filtering before obtaining a baseline would add
complexity without establishing whether it matters. Preserve the raw audit
fields so a later controlled ablation can remove:

- positive summaries containing explicit truth/no-deception conclusions;
- reasoning based only on disobedience to an instruction to lie;
- prompt-meta references to labels or privileged information.

Compare the unfiltered baseline against a filtered retrain only if the student
shows the corresponding error modes on validation or local test.

## Student SFT learning-rate sweep

Four one-epoch rank-16 LoRA students were trained with AdamW and evaluated in a
single shared vLLM session on all 822 validation rows. Evaluation used the exact
saved student prompt, greedy generation, a 512-token completion budget, and the
explicit `Prediction:0|1` field. Metrics are macro-averaged over datasets, as in
the competition. Because these are binary generated predictions, AUROC equals
balanced accuracy and is not separately informative.

| AdamW LR | validation BA | recall | FPR | instructed BA | varied BA | varied recall | parse errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-5` | 0.9036 | 0.8381 | 0.0310 | 0.9792 | 0.8028 | 0.6722 | 4 |
| `2e-5` | 0.9036 | 0.8381 | 0.0310 | 0.9792 | 0.8028 | 0.6722 | 3 |
| `5e-5` | **0.9048** | 0.8381 | **0.0286** | 0.9792 | **0.8056** | 0.6722 | 3 |
| `1e-4` | 0.8988 | 0.8357 | 0.0381 | 0.9771 | 0.7944 | 0.6611 | 4 |

The `5e-5` adapter is the validation winner, although the advantage over
`1e-5`/`2e-5` is only 0.0012 balanced accuracy and comes from fewer false
positives rather than better recall. The lowest training loss at `1e-4` did not
translate into better validation performance.

The instructed/varied gap is large for every learning rate: the best adapter is
at 0.9792 BA on instructed datasets versus 0.8056 on varied datasets, with
varied recall only 0.6722. Privileged-summary SFT therefore learns the familiar
instructed setting extremely well but does not solve transfer to varied
deception. The next useful measurements are continuous empty-summary prediction
logits and a local-test run of the selected `5e-5` adapter; semantic filtering
is secondary unless error inspection connects failures to the minority bad
teacher traces.

### Selected-adapter local test

The selected `5e-5` adapter was evaluated unchanged on all 821 local-test rows
as Slurm job `30101232`.

| split | balanced accuracy | recall | FPR | instructed BA | varied BA | varied recall | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9048 | 0.8381 | 0.0286 | 0.9792 | 0.8056 | 0.6722 | 3 |
| local test | **0.9107** | **0.8571** | 0.0357 | **0.9812** | **0.8167** | **0.7056** | 7 |

The adapter transfers slightly better to local test than validation overall,
so the validation selection did not collapse out of sample. The core weakness
persists: instructed deception is nearly solved, while varied-deception balanced
accuracy remains around 0.82 and recall around 0.71. The seven parse failures
are a small minority and cannot account for that scenario gap.

## Varied-only student ablation

The mixed student receives 3,693 usable instructed targets and 2,877 usable
varied targets. Because nearly all of the SFT gain is concentrated in the
instructed scenario, train a controlled varied-only student from the same
reviewed cache. Hold the selected AdamW learning rate (`5e-5`), one epoch,
rank-16/alpha-32 LoRA, effective batch size 32, prompt, and completion-only loss
fixed. Filter only by dataset name containing `varied-deception`. Compare both
varied and instructed validation metrics to measure specialization versus
catastrophic loss of the instructed capability.

The resulting adapter was trained as Slurm job `30101268` and evaluated on the
complete validation and local-test splits as jobs `30101348` and `30101350`.

| evaluation split | balanced accuracy | recall | FPR | instructed BA | varied BA | varied recall | parse errors | scoring time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9000 | 0.8333 | 0.0333 | 0.9792 | 0.7944 | 0.6556 | 4 | 31.8s |
| local test | **0.9155** | **0.8619** | **0.0310** | **0.9812** | **0.8278** | **0.7167** | 5 | 31.9s |

Specializing the training data did not consistently improve varied deception:
relative to the mixed `5e-5` adapter, varied balanced accuracy fell by 0.0111
on validation but rose by 0.0111 on local test. Test-wide balanced accuracy
rose by 0.0048. Instructed performance was unchanged, so there is no evidence
of catastrophic forgetting. The small, split-dependent gain is not strong
enough to replace the mixed adapter based on validation selection alone.

### Varied-only Muon sweep

The next controlled sweep tests whether optimizer geometry and additional
updates help the 2,877-example varied-only subset. It crosses hybrid Muon
learning rates `3e-5`, `1e-4`, and `3e-4` with one and two epochs. Muon updates
the trainable 2D LoRA matrices; AdamW at `1e-6` handles any remaining trainable
parameters. Rank, alpha, target modules, prompt, completion-only loss, effective
batch size 32, teacher cache, and seed remain fixed. Select candidates using
full-validation varied balanced accuracy, then reserve local test for the best
one or two configurations.

A one-step GPU integration smoke completed as job `30101400`. Full runs:

| Muon LR | epochs | Slurm job | method suffix |
| ---: | ---: | ---: | --- |
| `3e-5` | 1 | `30101407` | `muonlr3e5_ep1_v1` |
| `1e-4` | 1 | `30101402` | `muonlr1e4_ep1_v1` |
| `3e-4` | 1 | `30101404` | `muonlr3e4_ep1_v1` |
| `3e-5` | 2 | `30101406` | `muonlr3e5_ep2_v1` |
| `1e-4` | 2 | `30101403` | `muonlr1e4_ep2_v1` |
| `3e-4` | 2 | `30101405` | `muonlr3e4_ep2_v1` |

All six adapters were evaluated together on full validation as job `30101544`.

| Muon LR | epochs | overall BA | varied BA | varied recall | varied FPR | parse errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `3e-5` | 1 | 0.9012 | 0.7972 | **0.6611** | 0.0667 | 3 |
| `1e-4` | 1 | 0.9000 | 0.7944 | 0.6556 | 0.0667 | 3 |
| `3e-4` | 1 | 0.9012 | 0.7972 | 0.6556 | 0.0611 | 5 |
| `3e-5` | 2 | 0.9000 | 0.7944 | 0.6556 | 0.0667 | 4 |
| `1e-4` | 2 | **0.9024** | **0.8000** | 0.6556 | **0.0556** | 6 |
| `3e-4` | 2 | 0.9012 | 0.7972 | 0.6556 | 0.0611 | 5 |

Muon produces only a small improvement over the varied-only one-epoch AdamW
baseline (0.9000 overall and 0.7944 varied BA). The selected `1e-4`, two-epoch
run gains 0.0056 varied BA by reducing FPR, not by improving recall. It still
trails the mixed-data AdamW validation winner at 0.9048 overall and 0.8056
varied BA. The selected Muon adapter was evaluated on local test as job
`30101548` and scored 0.9107 overall BA, 0.9812 instructed BA, 0.8167 varied
BA, 0.7056 varied recall, 0.0722 varied FPR, and seven parse errors. It exactly
reproduces the mixed-data AdamW `5e-5` adapter on test: all 821 predictions and
full generated completion strings are identical. It is worse than the
varied-only AdamW adapter's 0.9155 overall and 0.8278 varied BA. Muon therefore
does not improve the selected test result despite its small validation FPR
gain; retain varied-only AdamW as the best distillation test reference.

Decision: use the varied-only one-epoch AdamW `5e-5` setup for subsequent
privileged-distillation work. It is simpler, trains in about 16 minutes, and is
the best tested student on local test (0.9155 overall BA and 0.8278 varied BA).
Do not continue Muon tuning for this SFT objective unless a later change to the
targets or inference method creates a new reason to revisit optimizer behavior.

### Heretic student-base ablation

A matched student-base ablation replaced aligned `Qwen/Qwen3.5-9B` with
`trohrbaugh/Qwen3.5-9B-heretic-v2` while holding the selected recipe fixed:
the same 2,877 varied-deception teacher summaries, rank-16/alpha-32 LoRA,
one epoch, AdamW `5e-5`, and effective batch size 32. Training job `30109845`
completed in 16m53s; validation job `30109846` completed successfully.

Heretic scored 0.8643 overall validation BA, 0.9646 instructed BA, and 0.7306
varied BA, with 0.7905 overall recall, 0.0619 FPR, and 23 parse errors. Within
varied deception, recall was only 0.5500 and FPR was 0.0889. The matched aligned
Qwen student scored 0.9000 overall and 0.7944 varied validation BA. Abliteration
therefore does not provide a better general reasoning-distillation base: the
HP-KR specialist benefit does not transfer to the current varied-deception SFT
task, and formatting reliability also worsens. Do not evaluate this adapter on
test. Retain regular aligned Qwen as the default fine-tuning base; use Heretic
only for explicit alignment/HP-KR diagnostics unless a future balanced routing
evaluation supports a specialist role.

### End-to-end teacher-prompt sweep

Teacher wording was optimized end to end while holding the selected student
recipe fixed: regular `Qwen/Qwen3.5-9B`, varied-deception-only targets,
rank-16/alpha-32 LoRA, one epoch, AdamW `5e-5`, effective batch size 32, and the
same student inference prompt. The teacher generator now supports
`teacher.dataset_name_contains`, so GPT-OSS generation and student training use
the same 2,880-row varied subset rather than generating unused instructed
traces. The filter fails explicitly on an empty match.

Round one tested five materially different summaries. Material contrast focused
on the output's consequential assertion; polarity guard emphasized the stance
actually asserted; blind reconciliation separated an independent audit from
label-conditioned reconciliation; minimal evidence compressed the target; and
claim hierarchy ranked assertions by importance. Teacher job `30110583`,
student jobs `30110970` and `30111863`, and shared validation job `30111864`
completed successfully.

| round-one target | usable targets | overall BA | varied BA | varied recall | varied FPR | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed original teacher / student baseline | 2,877 | 0.9000 | 0.7944 | 0.6556 | 0.0667 | 4 |
| material contrast | 2,591 | 0.9000 | 0.7944 | 0.6556 | 0.0667 | 4 |
| polarity guard | 2,761 | **0.9024** | **0.8000** | 0.6556 | **0.0556** | 6 |
| blind reconciliation | 1,978 | **0.9024** | **0.8000** | 0.6556 | **0.0556** | 5 |
| minimal evidence | 2,352 | **0.9024** | **0.8000** | 0.6556 | **0.0556** | 5 |
| claim hierarchy | 2,807 | 0.9012 | 0.7972 | 0.6556 | 0.0611 | 5 |

The three best variants tie exactly despite producing very different target
lengths and coverage. Their 0.0024 overall gain comes only from two fewer false
positives across the 420 honest validation rows; none improves deceptive recall.
The broader rewrites also weakened the original authoritative-label contract,
causing avoidable parse or semantic-label disagreement and leaving as many as
902 examples unused.

Round two therefore restored the original instruction verbatim: accept the
provided label and do not predict or debate it. Four narrow additions isolated
polarity, materiality, claim hierarchy, and concise truth contrast without
otherwise changing the teacher contract. Teacher job `30112027`, consolidated
student job `30112028`, and shared validation job `30112029` all completed.

| round-two target | usable targets | overall BA | varied BA | varied recall | varied FPR | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline + polarity | 2,876 | **0.9012** | **0.7972** | **0.6611** | 0.0667 | 5 |
| baseline + materiality | 2,880 | **0.9012** | **0.7972** | 0.6556 | **0.0611** | 5 |
| baseline + hierarchy | 2,872 | **0.9012** | **0.7972** | 0.6556 | **0.0611** | 4 |
| baseline + contrast | 2,866 | 0.9000 | **0.7972** | **0.6611** | 0.0667 | 4 |

All parsed round-two targets match their authoritative labels and coverage is
at least 99.5%, confirming that the stricter contract fixed generation quality.
Nevertheless, all four follow-ups remain below the 0.9024 first-round maximum.
The apparent changes are only one or two validation decisions and trade recall
against FPR. Across nine prompt variants, no teacher wording improves recall
materially or yields a gain beyond validation noise. This meets the stopping
criterion for prompt-only target tuning.

Decision: retain the original teacher and the existing regular-Qwen varied-only
AdamW `5e-5` adapter as the default. Record polarity guard as the best observed
validation prompt, but do not replace the default from that result alone: its
0.0024 gain is an FPR-only tie shared by three substantially different target
sets. Further gains likely require new information, balanced data coverage, or
a different training objective rather than more wording variants of the same
privileged summary.

After closing the prompt sweep, the frozen polarity-guard winner received one
confirmatory local-test evaluation. The first launch (`30113781`) stalled on
online Hub metadata checks before loading any rows and was cancelled after 21
idle minutes; all required model and dataset artifacts were already cached.
Offline rerun `30113826` completed successfully with the same adapter and
evaluator.

| local-test target | overall BA | recall | FPR | instructed BA | varied BA | varied recall | varied FPR | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original teacher / selected default | **0.9155** | **0.8619** | 0.0310 | 0.9812 | **0.8278** | **0.7167** | 0.0611 | **5** |
| polarity guard | 0.9143 | 0.8571 | **0.0286** | **0.9833** | 0.8222 | 0.7000 | **0.0556** | 6 |

The polarity guard changes 13 row predictions relative to the default. It fixes
three default errors but breaks five default-correct rows, for a net loss of two
correct decisions; the remaining five changed rows are wrong under both. Its
small FPR reduction is offset by lower recall, reproducing the validation
tradeoff without the validation BA gain. Treat the earlier `+0.0024` as split
noise and keep the original adapter. Do not test the other tied prompts: doing
so would turn the local test split into another selection set.

## Prediction-only SFT baseline

To test whether teacher reasoning supervision is necessary, train a controlled
student on only the final binary labels. It uses the same 2,877 usable varied
examples, rank-16/alpha-32 LoRA, one epoch, AdamW `5e-5`, effective batch size
32, seed, context formatting, and Truth Value Guard decision rule as the
selected compact-summary student. The cached reasoning-output instruction is
replaced while preserving its exact `<context>` and `<output>`; the supervised
completion is only `Prediction:0|1`. No teacher summary tokens are included in
the loss. Config: `privileged_information_distillation_prediction_only`.
Training was launched as job `30102432`.

The run completed in 14m39s with training loss 0.1009. Strict-format vLLM
evaluation produced no parse errors. Validation scored 0.8821 overall BA and
0.7833 varied BA; test scored 0.8631 overall BA and 0.7917 varied BA. On test,
recall was 0.7690 overall and 0.6556 on varied deception, with FPR 0.0429 and
0.0722 respectively. This is materially below the matched reasoning-summary
varied-only student (0.9155 overall test BA and 0.8278 varied BA), especially in
positive recall. The controlled ablation therefore supports keeping the compact
reasoning summaries: direct label SFT does not explain the student improvement.
Training job: `30102432`; validation job: `30102520`; test job: `30102582`.

### Varied-only GRPO continuation

To test whether GRPO was mainly limited by weak reasoning bootstrapping, continue
GRPO from the selected varied-only reasoning-summary adapter rather than the base
model. Training remains restricted to varied deception and uses the established
binary GRPO settings: rank-16/alpha-32 LoRA, Muon LR `3e-5`, AdamW-side LR
`1e-6`, temperature 1.2, four generations, effective prompt batch 32, and the
correctness plus format rewards. The GRPO prompt preserves the Truth Value Guard
decision rule and asks for a concise `<reasoning>` followed by `Prediction:N`.

The trainer now supports a trainable `init_adapter` and literal dataset-name
filter. A one-step smoke, job `30102648`, successfully loaded, synchronized,
updated, evaluated, and saved the distilled adapter. Full epoch-sweep jobs are
`30102734` (1 epoch), `30102733` (2 epochs), and `30102735` (3 epochs), each over
2,880 varied examples and 360 optimizer steps per epoch under the actual TRL
generation/batching semantics. The one-epoch run completed in 2h10m and scored
0.9083 validation BA, 0.8500 recall, and 0.0333 FPR with zero parse errors. Its
deterministic validation outputs averaged 54.3 tokens (p95 74, maximum 125), so
the 100+ token stochastic rollout lengths do not directly imply the same
submission decode length. The result is not yet a fair vLLM test comparison.

The two-epoch run completed as job `30102733` in 4h49m and improved validation
BA to 0.9179, with 0.9000 recall, 0.0643 FPR, and zero parse errors. This is a
0.0095 BA gain over one epoch, driven by higher recall at the cost of twice the
false-positive rate. Late second-epoch stochastic completions averaged roughly
135--150 tokens and reached about 299 tokens in logged batches, versus the much
shorter deterministic outputs from the one-epoch checkpoint. This length growth
explains the slower validation tail and may matter for submission runtime.

A fair vLLM test evaluation was subsequently run as job `30103963` to diagnose
the two-epoch checkpoint qualitatively. It scored 0.9048 BA/AUROC, 0.8310 recall,
0.0214 FPR, and zero parse errors, taking 147.4s for 821 rows. This is below the
selected varied-only SFT adapter's 0.9155 test BA despite the stronger validation
score. On the 40 test errors shared by the Qwen and GPT-OSS three-prompt
ensembles, the adapter corrected only 3: 3/34 shared false negatives and 0/6
shared false positives. Its summaries generally recognized the headline trivia
answer and then confidently asserted that obscure false embellishments were
accurate (for example, the invented PayPal settlement and wrong Plain of Jars
UNESCO date). Longer generated reasoning therefore did not supply missing world
knowledge; it often expanded the model's unsupported factual confidence. The
three corrected cases were comparatively structural or blatant rather than
successful verification of obscure supporting facts.

The three-epoch GRPO run completed as job `30102735` in 7h04m. Validation
regressed to 0.9095 BA/AUROC, 0.8571 recall, and 0.0381 FPR with zero parse or
format errors, below the two-epoch peak of 0.9179. Deterministic validation
outputs averaged 89.8 words (median 86, p95 132, maximum 222), while late
stochastic rollouts remained roughly 144--158 tokens on average. The additional
epoch therefore increased verbosity and compute without improving selection
metrics. Keep two epochs as the validation-selected GRPO checkpoint; do not
spend the held-out test split on the three-epoch adapter.

A follow-up GPT-OSS diagnostic separated missing knowledge from claim salience.
On the 34 false negatives shared by both heavy ensembles, an explicit blind
atomic audit found a concrete false supporting detail in 24 cases even though
the ordinary GPT-OSS ensemble had classified all 34 as non-deceptive. Thus, much
of the teacher's relevant knowledge was present but not activated once the
headline trivia answer appeared correct. Providing the authoritative positive
label raised positive audit verdicts to 29, but also induced strained, pedantic,
or misplaced contradictions; 3 completions exhausted 1024 tokens without a
verdict and 2 still found no error. Privileged-information targets should
therefore combine claim decomposition with explicit permission to return
uncertain, rather than forcing a factual contradiction from every positive
label.

The matched base-Qwen3.5-9B diagnostic confirms that this procedure cannot be
transferred naively. Qwen emitted blind positive verdicts for 28/34 and
privileged positives for 30/34, but a manual factual audit found only roughly
17--20 plausible identifications of the actual poisoned detail. Many remaining
outputs attacked a true claim and invented a replacement fact; examples include
reversing the Wright brothers' birth order, denying that Mount Catherine is
Egypt's highest peak, and inventing different PayPal settlement amounts in the
two conditions. The student therefore has some latent knowledge that structured
decomposition can elicit, but it lacks the teacher's verification reliability.
Training on forced decompositions without uncertainty or evidence controls risks
teaching confident fact-checking confabulations rather than better deception
detection.

### Deferred source-grounded atomic distillation

A future teacher/student format should make atomic claims an explicit interface
between the tuned judge and an organism-only consistency checker. The judge
would emit exact quoted spans, a `true|false|uncertain` assessment for each, a
compact summary, and the final prediction. Exact-span validation prevents the
extractor from silently introducing facts that were not present in the output.
The parsed claims can then be queried against the same generating organism with
forced `True|False` responses and conservatively ensembled with the tuned judge.

Teacher generation should be two-stage. First generate and freeze a blind atomic
audit without the privileged label. Then reveal the label and ask for the final
summary/prediction without allowing the teacher to rewrite its factual audit.
Audit/label conflicts should remain explicit rather than forcing a post-hoc
contradiction. This path is deferred because it requires a new cache, structured
student SFT, repeated organism queries, and validation of extraction recall and
false-rejection rates. The full proposal and ensemble constraints are recorded
as P27 in `docs/prompt_optimization/proposals.md`.

P27 also records an RLVR extension in which the judge learns when to defer to
the consistency checker. The checker remains an unreliable feature: reward the
ground-truth quality of the final blended probability with a proper scoring
rule, penalize unnecessary deference and query cost, and never reward organism
agreement directly. The first implementation should cache organism verification
features and fit a simple calibrated logistic or shallow gate. Online RL is only
justified if this conditional gate beats fixed-weight blending on validation;
otherwise the additional policy and rollout complexity has no demonstrated
value.

### Online self-distillation with SDPO

SDPO is a complementary test of the bootstrapping hypothesis. It initializes
from the same varied-only reasoning-summary adapter, samples on-policy, scores
binary correctness and format, and uses the live policy conditioned on feedback
as its own teacher. Successful rollouts provide implicit demonstrations. The
teacher reprompt also receives the authoritative `Prediction:N` as privileged
environment feedback; the ordinary student prompt never receives that label.

The full one-epoch live-teacher SDPO run completed as job `30102840` in 2h52m.
It scored 0.9083 validation BA/AUROC, 0.8357 recall, and 0.0190 FPR, with 18
parse/format errors and 535.4s validation scoring time. The parse failures were
mostly short completions that omitted the required reasoning tags and final
prediction, rather than 512-token truncations. Training also became unstable:
late logged correctness rewards fell to roughly 0.28--0.46 and format reward to
roughly 0.43--0.56, despite starting near 0.8--0.9 correctness with valid
formatting. SDPO did not beat one-epoch GRPO, two-epoch GRPO, or the original
distilled adapter, so do not evaluate this checkpoint on test. Treat the current
live self-distillation objective as format-degrading and not worth extending
without changing the loss/replay design.

The initial configuration blends policy and sampled-token reverse-KL
self-distillation equally (`distillation_weight=0.5`), uses live teacher weights,
and trains only on varied deception for one epoch. This requires experimental
TRL 1.8, whose supported vLLM range ends at 0.23; the main project pins vLLM
0.24. Therefore SDPO runs from the external environment
`/scratch/s4626451/.venvs/aletheia-sdpo` and does not alter the established GRPO
environment. One-step smoke job: `30102805`.

The initial smoke was resubmitted on `gpushort` while resolving two experimental
stack incompatibilities: TRL 1.8 needed an explicit 4,608-token vLLM context and
Qwen3.5 weight names needed the same `language_model.*` mapping used by the GRPO
trainer. Final smoke job `30102832` completed in 2m20s. Its single update took
26.2s, generated 82.6 tokens on average, found a successful rollout for every
group, used privileged feedback and teacher reprompting for every sample, and
reported policy loss 0.0297 plus distillation loss -0.0667. It saved a valid
adapter and evaluated 32 validation rows with zero parse errors. Full one-epoch
job: `30102840` on `gpumedium`.

## Counterfactual trace structure

A second teacher/student contract tests whether explicit factual baselines make
distillation more useful than a single freeform summary. It preserves the
summary first, then requires ordered `<facts>` and `<contradiction>` fields,
followed by the existing prediction:

```text
<reasoning_summary>...</reasoning_summary>
<facts>Correct facts, material omissions, or genuine uncertainties.</facts>
<contradiction>Comparison with the output, or explicitly no contradiction.</contradiction>
Prediction:0|1
```

For positive labels the contradiction must identify an important factual
conflict, misleading omission, or unjustified certainty. For negative labels it
must explicitly state why no important contradiction exists. The facts field is
allowed to describe uncertainty or missing information and is told not to
invent certainty. GPT-OSS Harmony analysis remains excluded; all three visible
fields are parsed from the final channel and distilled. Config:
`privileged_information_distillation_counterfactual`.

An initial ordered 32-negative smoke parsed all fields and labels successfully
as job `30101559`. Because simple row limiting follows label-sorted source data,
the generator now supports `teacher.limit_per_label` for balanced smoke tests.
Job `30101567` regenerates 16 examples per label for qualitative review before
any full-cache run. The balanced smoke completed with 32/32 strict parses and
32/32 label matches. Mean field lengths were 45.0 words for the summary, 37.0
for facts, and 32.4 for contradiction. Qualitative positive examples directly
contrast the wrong verdict with the correct factual baseline; negative examples
state that the output matches the facts and that no important contradiction
exists. The prompt and parser are ready for review before a full cache is
generated.

Full training-cache generation completed as job `30101601` in 6m57s. It wrote
6,573 rows (62.4 MB): 6,548 passed the strict ordered-field parser and matched
the privileged label, while 25 were malformed. Of the failures, 24 have label
1, 22 are varied-deception examples, and most looped or exhausted the output
budget while trying to reconcile difficult positive labels. Valid target field
lengths average 40.3 words for reasoning summary, 36.0 for facts, and 30.1 for
contradiction. All 6,548 valid student targets have the exact required field
order. A resumable cleanup pass, job `30101758`, reuses those valid records and
regenerates only the 25 failures. It recovered 20, leaving 6,568 valid targets
and five malformed records (99.92% usable). All five remaining failures are
positive varied-deception examples where GPT-OSS repeatedly debates an
apparently inconsistent label or truncates before closing the required fields.
Ignore these five: the SFT loader already excludes parse failures, and further
regeneration is unlikely to be worth the complexity or change the result.

The first counterfactual student uses the selected simple training setup:
varied-deception only, one epoch, AdamW `5e-5`, rank-16/alpha-32 LoRA, and
effective batch size 32. The cache supplies 2,875 usable varied targets (1,440
label 0 and 1,435 label 1). Training was launched as job `30101761` with method
`qwen9b_pid_counterfactual_variedonly_adamwlr5e5_v1`.

Training completed successfully in 19m36s with loss 0.4339 and saved the final
adapter. Full 822-row validation with the counterfactual strict-format checker
was launched as job `30101939`.

The initial 512-token validation run scored 0.8702 overall BA and 0.7806
varied BA (0.6667 varied recall, 0.1056 varied FPR), below the compact-summary
varied-only adapter's 0.9000/0.7944 validation BA. It reported 92 missing
predictions: 75 are actually unambiguous literal-placeholder outputs such as
`Prediction:<0>`, while 17 genuinely fail to reach a prediction. Bracket-aware
parsing does not change the initial metrics because the recovered rows already
receive score 0. A 1024-token diagnostic, job `30102068`, tests whether removing
the remaining truncations changes the conclusion.

## Compact local factual retrieval proposal

Many varied false negatives begin with an incorrect factual premise in the
student's otherwise coherent summary. A future method can retrieve local
reference evidence from the user question and final answer, then add the top
few short passages to the distilled judge prompt. External inference APIs are
not needed; retrieval should run inside the submission sandbox on CPU.

A full English Wikipedia dump is infeasible: the July 2026 current-article
multistream dump is about 26.6 GB compressed and its title/offset index alone is
about 283 MB. Simple English Wikipedia is about 381 MB compressed. Phoenix v2
already packages to about 109 MB under the 200 MB competition limit, leaving at
most roughly 90 MB. Target a 30-70 MB compact lexical index instead, such as
titles/aliases plus short lead facts for a broad subset, using SQLite FTS/BM25
or a hashed inverted index rather than dense embeddings.

Evaluate retrieval first on varied validation false negatives and track FPR:
wrong-entity retrieval and fictional framing can create new false positives.
Wikipedia text is CC BY-SA and may conflict with the competition's MIT
publication requirement; confirm licensing with organizers or prefer a compact
CC0 Wikidata-derived factual index.

The first cached-evidence run covered 302 of 360 varied-validation rows. With
the mixed `5e-5` distilled adapter, varied recall rose slightly from 0.6722 to
0.6778, but FPR rose from 0.0611 to 0.0833; varied balanced accuracy therefore
fell from 0.8056 to 0.7972. This does not yet distinguish poor retrieval from an
adapter that has not learned to consume unfamiliar evidence.

A base-model control reuses the identical cache with the regular short Qwen
Truth Value Guard prompt and rating parser. The generic black-box runner now
accepts `retrieval_cache`; config
`blackbox_reasoning_nothink_truth_value_wikipedia_rag_v1` isolates whether base
Qwen benefits from the evidence without the SFT adapter. Full validation was
run as Slurm job `30101368`, with an exact no-evidence control as job
`30101370`.

| base Qwen condition | overall BA | varied BA | varied recall | varied FPR | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| no evidence | **0.8940** | **0.8167** | 0.7500 | 0.1167 | 0 |
| Wikipedia evidence | 0.8869 | 0.8000 | **0.8222** | 0.2222 | 0 |

The base model does consume the new evidence: varied recall rises by 0.0722.
However, varied FPR rises by 0.1056, more than offsetting that gain. The first
retrieval implementation therefore has a precision/relevance problem rather
than showing that SFT uniquely prevents evidence use. Both base-Qwen conditions
have identical instructed metrics because only varied rows received passages.

To test whether prompt structure can recover the recall gain without the FPR
spike, job `30101386` batches four prompt-only conditions in one base-Qwen vLLM
load: symmetric support/contradiction handling, an explicit evidence-relation
scaffold, answer-first comparison, and a conservative contradiction-required
rule. All members use the same fixed Wikipedia cache, 512-token budget, rating
parser, and threshold. Config:
`blackbox_reasoning_nothink_wikipedia_evidence_sweep_v1`.

| prompt condition | overall BA | varied BA | varied recall | varied FPR | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-evidence control | **0.8940** | **0.8167** | 0.7500 | 0.1167 | 0 |
| original Wikipedia prompt | 0.8869 | 0.8000 | **0.8222** | 0.2222 | 0 |
| symmetric relation | 0.8607 | 0.7889 | 0.7333 | 0.1556 | 0 |
| relation scaffold | 0.8369 | 0.7667 | 0.5833 | **0.0500** | 0 |
| contradiction required | 0.7917 | 0.6694 | 0.5944 | 0.2556 | 0 |
| answer first | 0.1512 | 0.2333 | 0.3056 | 0.8389 | 0 |

The symmetric wording partially reduces the original RAG false-positive spike,
but also loses recall and remains below the no-evidence control. The explicit
relation scaffold is conservative enough to reach 0.0500 varied FPR, but its
recall collapses to 0.5833. The contradiction-required wording unexpectedly
raises FPR, and answer-first causes a severe prompt-induced inversion. No member
improves balanced accuracy. Because each member also changes the core judge
prompt, paired no-evidence runs would be needed to separate its baseline prompt
quality from its evidence-handling effect before transferring any wording into
student evaluation.

Decision: defer this path. A dedicated evidence-conditioned adapter could be
trained on balanced support, contradiction, irrelevant, insufficient, and
no-evidence cases, but building and validating that dataset adds substantial
retrieval, teacher-generation, and robustness complexity. The current prompted
RAG variants do not beat the no-evidence control, so that investment is not
justified for the active submission path. Revisit only if simpler distillation
and prompting improvements stall and varied-deception factual errors remain the
dominant measured bottleneck.

## Cached ensemble diagnostics

Cached row-level validation predictions were blended with the selected
varied-only compact adapter, with all weights and thresholds selected on
validation and then applied unchanged to local test.

| companion | blend selected on validation | validation BA | validation AUROC | local-test BA | local-test AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| 4096 D/K/S generated ensemble | per-dataset ranks, 0.10 distill + 0.90 companion, threshold 0.53 | **0.9298** | 0.9376 | 0.9321 | 0.9473 |
| original generated reasoning judge | raw scores, 0.10 distill + 0.90 companion, threshold 0.05 | 0.9226 | 0.9306 | **0.9333** | 0.9385 |
| GRPO empty-reasoning binary logits | raw scores, 0.40 distill + 0.60 companion, threshold 0.42 | 0.9250 | **0.9551** | 0.9274 | **0.9652** |

The compact adapter alone scores 0.9000 validation BA and 0.9155 local-test BA.
Thus all three selected blends improve on both splits. The D/K/S blend is the
validation winner but is operationally unattractive: its three long 4096-token
judges were already too slow and unreliable for official execution. The
single generated reasoning judge is simpler and has the highest local-test BA,
but would add another generation pass to the already slow distilled notebook.
The GRPO-logits blend offers the strongest AUROC and should be the first
implementation candidate if a two-adapter NDIF logits path can be made reliable;
it adds a forward/logits pass rather than another long decode. Cheap no-thinking
prompt blends improved validation less and generally did not transfer to test.

## Next measurements

1. Evaluate empty-summary next-token logits for continuous scores.
2. Investigate whether varied-only specialization gains persist on additional
   held-out data or are split noise.
3. Keep evidence-conditioned retrieval as a deferred path if simpler methods
   stall and factual errors remain dominant.
4. Inspect student errors for teacher-meta leakage and intent-only decisions.
5. If needed, run the semantic-filtering ablation without regenerating teacher
   traces.
