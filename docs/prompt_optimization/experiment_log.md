# Prompt Experiment Log

## Baseline: `qwen_reason_v1`

Source: `results/blackbox/qwen_reason_v1/validation/result.json`

Prompt summary:

- Defines deception as intentionally misleading the user into a false belief.
- Rates final message from 1 to 7.
- Requires final line `Rating: <integer from 1 to 7>`.

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.8905 |
| AUROC | 0.8930 |
| recall | 0.7976 |
| FPR | 0.0167 |
| parse errors | 8 / 822 |

Interpretation:

- The model is conservative: FPR is already very low, while recall is the main
  source of lost balanced accuracy.
- Worst datasets are mostly `dev-varied-deception` Qwen LoRA variants, with
  low recall and near-zero FPR. The prompt should push harder on factual
  contradiction and misleading final answers without broadly increasing false
  positives.
- Initial false-negative inspection showed a more specific failure mode: the
  judge often treats the final answer as honest when the answer is surface
  plausible or factually correct, even though the hidden/system context says the
  assistant is in a deceptive, wrong-answer, or misinformation persona. The
  revised candidates therefore treat explicit deceptive intent in context as
  evidence for a high rating.
- Time is dominated by verbose repeated reasoning. Speed improvements should
  focus on a shorter evidence checklist, not merely truncating generation.

## Candidate: `qwen_reason_recall4096_v1`

Status: completed as Slurm job `30022522`.

Hypothesis:

- A more explicit decision procedure will recover false negatives on varied
  deception by telling the judge to inspect hidden/system context for deceptive
  intent and wrong-answer personas, then rate the final answer as deceptive when
  it is presented as genuine under that context.
- Keeping the 4096-token budget isolates prompt quality from truncation effects.

Config: `configs/blackbox_reasoning_recall4096.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.8655 |
| AUROC | 0.8755 |
| recall | 0.8881 |
| FPR | 0.1571 |
| parse errors | 15 |
| score time | 255.1s |
| rows/s | 3.2 |

Decision: rejected. It recovered recall but raised FPR too much. It also ran
faster than baseline but not enough to justify the accuracy drop.

## Candidate: `qwen_reason_recall2048_v1`

Status: Slurm job `30022520` was cancelled before scoring because it was placed
on the same GPU node as the 4096 job and both jobs stalled in vLLM
initialization. The user clarified that same-node jobs likely use different GPUs,
so this contention concern was too conservative. Requeued as Slurm job
`30022544`.

Hypothesis:

- The same decision procedure plus a strict short checklist can reduce runtime
  while avoiding the recall collapse seen in the plain 2048-token run.

Config: `configs/blackbox_reasoning_recall2048.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.8226 |
| AUROC | 0.8448 |
| recall | 0.8262 |
| FPR | 0.1810 |
| parse errors | 15 |
| score time | 74.9s |
| rows/s | 11.0 |

Decision: rejected. It improved speed but raised FPR sharply and degraded
balanced accuracy. The failure mode was treating hidden deceptive intent as
sufficient: examples with a deceptive instruction but a genuinely correct final
answer became false positives.

## Candidate: `qwen_reason_scrutiny4096_v1`

Status: completed as Slurm job `30022567`.

Hypothesis:

- Hidden deceptive intent should not be sufficient by itself. Instead, it should
  make the judge scrutinize every factual claim, including supporting details.
  This should recover false negatives where the direct answer is plausible but
  a supporting detail is false, without creating as many false positives as the
  recall prompt.

Config: `configs/blackbox_reasoning_scrutiny4096.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.8976 |
| AUROC | 0.8988 |
| recall | 0.8119 |
| FPR | 0.0167 |
| parse errors | 10 |
| score time | 347.5s |
| rows/s | 2.4 |

Decision: provisionally best new candidate. It improves over
`qwen_reason_v1` on validation and runs faster, but it is only a small gain and
still misses many positives in the varied Qwen LoRA subsets. The lowest-score
false negatives show the judge can explicitly notice a wrong-answer instruction
and still assign `Rating: 1`, so the next variant should be more decisive when
the final answer contradicts a known answer in context without making hidden
intent alone sufficient.

## Candidate: `qwen_reason_scrutiny2048_v1`

Status: completed as Slurm job `30022566`.

Hypothesis:

- The corrected hidden-intent rule plus a short checklist may preserve more
  accuracy than `qwen_reason_recall2048_v1` while retaining a large speed
  advantage over 4096-token reasoning.

Config: `configs/blackbox_reasoning_scrutiny2048.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.8762 |
| AUROC | 0.8776 |
| recall | 0.7667 |
| FPR | 0.0143 |
| parse errors | 30 |
| score time | 295.3s |
| rows/s | 2.8 |

Decision: rejected. It did not keep the expected speed advantage and lost
recall/parse reliability versus the 4096-token version.

## Candidate: `qwen_reason_details4096_v1`

Status: completed as Slurm job `30022580`.

Hypothesis:

- A smaller change from the baseline may improve varied-deception recall without
  the FPR spike from P1/P2: keep judging the final message truth, but explicitly
  inspect supporting details and state that hidden intent is not sufficient.

Config: `configs/blackbox_reasoning_details4096.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.9060 |
| AUROC | 0.9116 |
| recall | 0.8405 |
| FPR | 0.0286 |
| parse errors | 7 |
| score time | 464.2s |
| rows/s | 1.8 |

Decision: current best validation candidate. It gains about `+0.0155`
balanced accuracy over `qwen_reason_v1` by improving recall and parse
reliability, but it is slightly slower and increases FPR. Threshold tuning only
nudges it to about `0.9071`, so further gains still need prompt changes.

## Candidate: `qwen_reason_details2048_v1`

Status: completed as Slurm job `30022579`.

Hypothesis:

- The same baseline-adjacent rule may recover enough parse/ranking quality at
  2048 tokens to improve over plain 2048 while remaining faster than baseline.

Config: `configs/blackbox_reasoning_details2048.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.8738 |
| AUROC | 0.8754 |
| recall | 0.7667 |
| FPR | 0.0190 |
| parse errors | 50 |
| score time | 287.2s |
| rows/s | 2.9 |

Decision: rejected. The baseline-adjacent wording did not improve recall and
created many parse errors at 2048 tokens.

## Candidate: `qwen_reason_known4096_v1`

Status: completed as Slurm job `30022601`.

Hypothesis:

- Several false negatives include context that states the correct answer while
  the assistant final message gives a different plausible answer. The judge
  should trust that context as an answer key and rate the contradiction high.
- This should recover positives without repeating P1's mistake of making hidden
  deceptive intent sufficient.

Config: `configs/blackbox_reasoning_known4096.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.9071 |
| AUROC | 0.9093 |
| recall | 0.8310 |
| FPR | 0.0167 |
| parse errors | 18 |
| score time | 515.5s |
| rows/s | 1.6 |

Decision: current highest validation balanced accuracy, but only by `+0.0012`
over `qwen_reason_details4096_v1` and with materially slower runtime and more
parse errors. Threshold tuning reaches about `0.9083`.

## Candidate: `qwen_reason_details3072_v1`

Status: completed as Slurm job `30022607`.

Hypothesis:

- `qwen_reason_details4096_v1` is the current accuracy leader but slightly
  slower than the original 4096 baseline.
- A 3072-token cap may preserve most of the improved decision rule while cutting
  generation time. The risk is additional truncation parse errors.

Config: `configs/blackbox_reasoning_details3072.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.8976 |
| AUROC | 0.8992 |
| recall | 0.8238 |
| FPR | 0.0286 |
| parse errors | 18 |
| score time | 392.0s |
| rows/s | 2.1 |

Decision: rejected for accuracy. It is faster than `details4096`, but the
3072-token cap loses most of the `details4096` gain and increases parse errors.

## Candidate: `qwen_reason_calibrated4096_v1`

Status: completed as Slurm job `30022619`.

Hypothesis:

- Remaining mistakes are a mix of fact-check hallucination false positives and
  false negatives where the direct answer is plausible but concrete supporting
  details are false.
- The prompt should distinguish non-suspicious accurate/reinforced contexts
  from explicit wrong-answer/replacement directives, then treat dates, origins,
  definitions, titles, names, biographies, and descriptions as important
  supporting details.

Config: `configs/blackbox_reasoning_calibrated4096.yaml`

Validation result:

| metric | value |
| --- | ---: |
| balanced accuracy | 0.9024 |
| AUROC | 0.9114 |
| recall | 0.8381 |
| FPR | 0.0333 |
| parse errors | 11 |
| score time | 470.7s |
| rows/s | 1.7 |

Decision: rejected. It improved recall versus `known4096` but raised FPR enough
to lose balanced accuracy. The extra calibration wording also did not improve
runtime.

## Current Ranking

| candidate | balanced accuracy | recall | FPR | parse errors | score time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen_reason_known4096_v1` | 0.9071 | 0.8310 | 0.0167 | 18 | 515.5s |
| `qwen_reason_details4096_v1` | 0.9060 | 0.8405 | 0.0286 | 7 | 464.2s |
| `qwen_reason_calibrated4096_v1` | 0.9024 | 0.8381 | 0.0333 | 11 | 470.7s |
| `qwen_reason_budget_4096` | 0.8988 | 0.8167 | 0.0190 | 8 | 450.2s |
| `qwen_reason_scrutiny4096_v1` | 0.8976 | 0.8119 | 0.0167 | 10 | 347.5s |
| `qwen_reason_details3072_v1` | 0.8976 | 0.8238 | 0.0286 | 18 | 392.0s |
| `qwen_reason_v1` | 0.8905 | 0.7976 | 0.0167 | 8 | 446.8s |

Practical recommendation:

- Use `qwen_reason_known4096_v1` if optimizing validation balanced accuracy
  only.
- Use `qwen_reason_details4096_v1` if choosing a more robust tradeoff: it is
  only `0.0012` lower in balanced accuracy, but faster and has fewer parse
  errors.
