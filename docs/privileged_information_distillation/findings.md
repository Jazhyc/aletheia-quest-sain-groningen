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
varied BA. The selected Muon adapter was launched on local test as job
`30101548`.

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

## Next measurements

1. Evaluate empty-summary next-token logits for continuous scores.
2. Investigate whether varied-only specialization gains persist on additional
   held-out data or are split noise.
3. Keep evidence-conditioned retrieval as a deferred path if simpler methods
   stall and factual errors remain dominant.
4. Inspect student errors for teacher-meta leakage and intent-only decisions.
5. If needed, run the semantic-filtering ablation without regenerating teacher
   traces.
