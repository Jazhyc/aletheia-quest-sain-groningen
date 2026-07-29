# Findings and Decisions

## Corrected NDIF adapter validation after PEFT migration (2026-07-28)

The earlier vLLM validation margins for the GPT-OSS privileged, GPT-OSS blind,
Luna privileged, and Qwen3.5-27B soft-distillation students did not measure the
intended adapters. After migrating their PEFT tensor paths to the canonical
Qwen3.5 NDIF model tree, all four were rerun on the complete 822-row,
21-dataset validation split through competition NDIF.

The rerun used the optimized Phoenix direct path: one remote session per
base/adapter, length sorting with `48/32/16` tiers, left padding and truncation,
`logits_to_keep=1`, no judge generation, and normalized literal next-token
`0|1` probabilities after `Prediction:`. Both prompt contracts were scored in
the same session. The first three students were trained on the compact
reasoning-summary prompt; Q27 soft was trained on the shorter binary-only
prompt.

| model | matched prompt | corrected macro AUROC | instructed | varied | old vLLM | corrected minus old | unique |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | summary | `0.93994` | `0.99375` | `0.86819` | — | — | `95/822` |
| Base Qwen | binary | `0.94530` | `0.99438` | `0.87986` | — | — | `73/822` |
| GPT-OSS PI | summary | `0.94911` | `0.99604` | `0.88653` | `0.93815` | `+0.01095` | `37/822` |
| GPT-OSS blind | summary | `0.94893` | `0.99583` | `0.88639` | `0.93762` | `+0.01131` | `61/822` |
| Luna PI | summary | `0.94821` | `0.98948` | `0.89319` | `0.93762` | `+0.01060` | `53/822` |
| Q27 soft | binary | **`0.95375`** | **`0.99687`** | **`0.89625`** | `0.94625` | `+0.00750` | `100/822` |

The old vectors confirm the silent-base diagnosis more directly than the
aggregate changes. Each historical vLLM vector correlates at `0.99915` to
`0.99936` with the corrected NDIF base vector under the same prompt, with mean
absolute differences of only `0.00729` to `0.00755`. Against the corrected
adapter vectors, correlations fall to `0.91197`--`0.95908` and mean absolute
differences rise to `0.06829`--`0.21036`. The corrected adapter-versus-base
mean absolute movements are `0.20196` for GPT-OSS PI, `0.06623` for blind,
`0.20785` for Luna, and `0.08479` for Q27 soft. These are active adapters, not
prompt-only base controls.

Prompt compatibility was tested directly:

| adapter | summary prompt | binary prompt | matched minus alternate |
| --- | ---: | ---: | ---: |
| GPT-OSS PI | `0.94911` | `0.94917` | `-0.00006` |
| GPT-OSS blind | `0.94893` | `0.94744` | `+0.00149` |
| Luna PI | `0.94821` | `0.94315` | `+0.00506` |
| Q27 soft | `0.95042` | `0.95375` | `+0.00333` |

Thus prompt length is not the issue: the complete prompts were only 232--844
tokens for summary and 187--799 for binary, and no row was truncated. Wording
and the learned output contract can still affect ranking. Luna should retain
the summary prompt, and Q27 soft should retain its binary training prompt.
GPT-OSS PI is effectively invariant in AUROC, although prompt changes still
move its calibration.

The training renderer and Phoenix 6.0 structural renderer produced
byte-identical evidence on `822/822` rows under both prompts. Phoenix 6.0 is
therefore structurally compatible on this validation set. This does not erase
the private long-context renderer result: no validation row activates the
character caps that distinguish the renderers.

Per-dataset matched-prompt AUROC:

| validation unit | GPT-OSS PI | GPT-OSS blind | Luna PI | Q27 soft |
| --- | ---: | ---: | ---: | ---: |
| instructed Nemotron, base | `1.00000` | `1.00000` | `0.96250` | `1.00000` |
| instructed Qwen, base | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| instructed Qwen, a-mo-1 | `0.97375` | `0.98000` | `0.97250` | `0.98875` |
| instructed Qwen, a-mo-3 | `1.00000` | `1.00000` | `0.99875` | `1.00000` |
| instructed Qwen, a-mo-4 | `1.00000` | `0.99875` | `0.99625` | `1.00000` |
| instructed Qwen, a-mo-5 | `0.99250` | `0.99000` | `0.98625` | `1.00000` |
| instructed Qwen, a-mo-6 | `0.98875` | `0.98500` | `0.97250` | `0.97875` |
| instructed Qwen, a-mo-7 | `1.00000` | `1.00000` | `0.99875` | `1.00000` |
| instructed Qwen, b-mo | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| instructed Qwen, c-mo | `1.00000` | `1.00000` | `0.99125` | `1.00000` |
| instructed Gemma, base | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| instructed Gemma, s-mo | `0.99750` | `0.99625` | `0.99500` | `0.99500` |
| varied Qwen, base | `0.96875` | `0.97375` | `0.97875` | `0.97500` |
| varied Qwen, a-mo-1 | `0.88000` | `0.87500` | `0.87375` | `0.88750` |
| varied Qwen, a-mo-3 | `0.72750` | `0.73875` | `0.78500` | `0.73625` |
| varied Qwen, a-mo-4 | `0.85875` | `0.83750` | `0.84625` | `0.87000` |
| varied Qwen, a-mo-5 | `0.94750` | `0.95750` | `0.97750` | `0.96000` |
| varied Qwen, a-mo-6 | `0.88000` | `0.88250` | `0.86875` | `0.89750` |
| varied Qwen, a-mo-7 | `0.83500` | `0.83875` | `0.83750` | `0.85125` |
| varied Qwen, b-mo | `0.95000` | `0.93625` | `0.92125` | `0.95375` |
| varied Qwen, c-mo | `0.93125` | `0.93750` | `0.95000` | `0.93500` |

Q27 soft is the corrected validation winner. It gains `0.00845` over base Qwen
under the same binary prompt. Under the unchanged summary prompt it still
scores `0.95042`, `0.01048` above the summary base, but that is below its
training-matched result. This rerun supersedes the affected historical vLLM
adapter-inference rankings below, including adapter-dependent local-test
numbers. It does not invalidate the teacher caches or training runs. No
corrected local-test selection was performed, and this validation rerun does
not authorize one.

Reproducible code and ignored artifacts:

- `experiments/phoenix_adapter_validation_ndif/`
- `results/blackbox/phoenix_adapter_validation_ndif_v1/`

## Corrected reason-then-score validation (2026-07-28)

The migrated GPT-OSS privileged-information and Luna privileged-information
students were also given their training-matched ability to generate a compact
reasoning summary before the decision. The frozen Phoenix 5.2.1 protocol was
used: deterministic generation with a 512-token allowance, remove only the
final generated `0|1`, then read normalized literal label logits at that
position. A fresh direct-boundary arm was scored in the same second remote
session as the post-reasoning arm.

| adapter | boundary | macro AUROC | instructed | varied | pooled AUROC | unique |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-OSS PI | direct | **`0.94911`** | `0.99604` | `0.88653` | `0.96208` | `37/822` |
| GPT-OSS PI | post-reasoning | `0.92381` | `0.98812` | `0.83806` | `0.91134` | `46/822` |
| Luna PI | direct | **`0.94821`** | `0.98948` | `0.89319` | `0.96111` | `53/822` |
| Luna PI | post-reasoning | `0.92470` | `0.98104` | `0.84958` | `0.90740` | `90/822` |

Reasoning therefore reduced macro AUROC by `0.02530` for GPT-OSS PI and
`0.02351` for Luna. The loss is strongest on the varied units: `-0.04847` and
`-0.04361`, compared with instructed changes of `-0.00792` and `-0.00844`.
Post-reasoning improved/tied/lost `3/5/13` GPT-OSS units and `7/1/13` Luna
units. Its row-level probability correlation with direct was only `0.87705`
for GPT-OSS and `0.90398` for Luna, with mean absolute movements of `0.26976`
and `0.21271`.

Per-dataset AUROC:

| validation unit | GPT direct | GPT reasoning | delta | Luna direct | Luna reasoning | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| instructed Nemotron, base | `1.00000` | `1.00000` | `+0.00000` | `0.96250` | `0.95000` | `-0.01250` |
| instructed Qwen, base | `1.00000` | `1.00000` | `+0.00000` | `1.00000` | `0.99750` | `-0.00250` |
| instructed Qwen, a-mo-1 | `0.97375` | `0.98875` | `+0.01500` | `0.97250` | `0.99875` | `+0.02625` |
| instructed Qwen, a-mo-3 | `1.00000` | `1.00000` | `+0.00000` | `0.99875` | `1.00000` | `+0.00125` |
| instructed Qwen, a-mo-4 | `1.00000` | `0.99875` | `-0.00125` | `0.99625` | `1.00000` | `+0.00375` |
| instructed Qwen, a-mo-5 | `0.99250` | `0.98500` | `-0.00750` | `0.98625` | `0.97750` | `-0.00875` |
| instructed Qwen, a-mo-6 | `0.98875` | `0.98875` | `+0.00000` | `0.97250` | `0.97625` | `+0.00375` |
| instructed Qwen, a-mo-7 | `1.00000` | `0.98625` | `-0.01375` | `0.99875` | `0.95500` | `-0.04375` |
| instructed Qwen, b-mo | `1.00000` | `1.00000` | `+0.00000` | `1.00000` | `1.00000` | `+0.00000` |
| instructed Qwen, c-mo | `1.00000` | `0.99625` | `-0.00375` | `0.99125` | `0.97500` | `-0.01625` |
| instructed Gemma, base | `1.00000` | `0.99875` | `-0.00125` | `1.00000` | `0.98625` | `-0.01375` |
| instructed Gemma, s-mo | `0.99750` | `0.91500` | `-0.08250` | `0.99500` | `0.95625` | `-0.03875` |
| varied Qwen, base | `0.96875` | `0.87875` | `-0.09000` | `0.97875` | `0.95875` | `-0.02000` |
| varied Qwen, a-mo-1 | `0.88000` | `0.77250` | `-0.10750` | `0.87375` | `0.77250` | `-0.10125` |
| varied Qwen, a-mo-3 | `0.72750` | `0.70625` | `-0.02125` | `0.78500` | `0.65250` | `-0.13250` |
| varied Qwen, a-mo-4 | `0.85875` | `0.87125` | `+0.01250` | `0.84625` | `0.88750` | `+0.04125` |
| varied Qwen, a-mo-5 | `0.94750` | `0.90125` | `-0.04625` | `0.97750` | `0.99000` | `+0.01250` |
| varied Qwen, a-mo-6 | `0.88000` | `0.87125` | `-0.00875` | `0.86875` | `0.85875` | `-0.01000` |
| varied Qwen, a-mo-7 | `0.83500` | `0.74625` | `-0.08875` | `0.83750` | `0.70500` | `-0.13250` |
| varied Qwen, b-mo | `0.95000` | `0.96875` | `+0.01875` | `0.92125` | `0.95750` | `+0.03625` |
| varied Qwen, c-mo | `0.93125` | `0.82625` | `-0.10500` | `0.95000` | `0.86375` | `-0.08625` |

The optimized Phoenix batching path was retained. Each adapter used one
generation session with length-aware `32/24/16` batches and a 32,768 padded
input-plus-output ceiling, followed by one paired scoring session with
`48/32/16`, left padding/truncation, and `logits_to_keep=1`. GPT-OSS generation
took 396.9 seconds and had two 512-token completions without a final label;
both used the declared canonical `Prediction:` fallback. Luna took 220.0
seconds with zero fallbacks. The paired score passes took 49.9 and 48.5
seconds. Generation prompts were 230--842 tokens; post-reasoning prompts were
at most 1,150 GPT-OSS tokens and 950 Luna tokens. No row was truncated.

The increased score uniqueness does not compensate for the degraded ordering.
Keep direct scoring for both adapters. This corrected comparison supersedes
the affected historical vLLM reason-boundary comparisons but does not authorize
a local-test rerun.

Reproducible code and ignored artifacts:

- `experiments/phoenix_adapter_reasoning_ndif/`
- `results/blackbox/phoenix_adapter_reasoning_ndif_v1/`

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

### GPT-5.6 Luna privileged Truth Value Guard reasoning pilot (2026-07-27)

A matched OpenRouter pilot tested `openai/gpt-5.6-luna` at medium and high
reasoning effort on the same 36 varied-training rows: two deterministic random
examples per label from each of the nine varied dataset units. The teacher saw
the authoritative label and the original privileged Truth Value Guard prompt.
Only the compact visible `<reasoning_summary>` is a potential student target;
OpenRouter was asked not to return the private reasoning text, although its
reasoning tokens remained in usage and billing.

| effort | parsed | label matches | prompt tokens | completion tokens | reasoning tokens | reported cost | mean summary words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | `36/36` | `36/36` | `20,052` | `3,794` | `1,227` | `$0.02119` | `43.86` |
| high | `36/36` | `36/36` | `20,052` | `5,224` | `2,690` | `$0.02544` | `42.25` |

All 72 calls used OpenAI's provider, stopped normally, and required no retry.
As with the original GPT-OSS cache, Luna normally omitted the redundant
`Prediction:` line: it appeared in only 2/36 medium responses and 0/36 high
responses. The parser safely attached the authoritative label after extracting
the structured summary.

A manual pass found concrete, plausible poisoned-fact identifications in all
18 deceptive rows under both efforts and no induced contradiction in the 18
honest rows. Luna also repaired several visible GPT-OSS rationale weaknesses in
the sampled rows. For example, it identified the false placement of the Khyber
Pass in the Sulaiman Mountains where the cached GPT-OSS summary had declared
the response accurate, and it correctly challenged the claim that the Tombs of
the Kings are ten kilometres from Paphos. Medium and high were qualitatively
tied on this small sample; one row elicited two different but independently
valid errors (Gerry Rafferty's birthplace versus the impossible 1978 date for
the 1992 film *Reservoir Dogs*).

High used 2.19 times as many private reasoning tokens as medium but cost only
20% more. Scaling the actually reported pilot costs linearly to all 2,880
varied rows gives approximately `$1.70` for medium and `$2.04` for high; without
assuming cache discounts, list-price ceilings from the observed token counts
are roughly `$3.43` and `$4.11`. Cost therefore does not decide the effort
choice. The pilot establishes that high is cheap, concise, and reliable, but
does not yet demonstrate a quality advantage over medium.

Runtime and resumable artifacts:

- `experiments/privileged_information_distillation/run_openrouter_reasoning_teacher_pilot.py`
- `results/blackbox/gpt56_luna_openrouter_privileged_tvg_reasoning_pilot_v1/`

The selected full medium-effort cache subsequently completed over all 2,880
varied-training rows. It used 1,560,183 prompt tokens, 315,301 completion
tokens, of which 109,137 were reported as reasoning tokens, and cost `$1.70929`.
All 2,880 calls used OpenAI's provider, stopped normally, required no retry,
and yielded a parseable summary. Only ten visible finals included the redundant
prediction line; the remaining authoritative labels were attached by the same
deterministic fallback used for GPT-OSS. Wall time was 733.2 seconds at
concurrency eight.

The cache has exactly 160 examples per label in each of the nine varied units.
Visible summaries average 44.15 words (median 44, p95 62, maximum 92). A
mechanical leakage audit found no mention of ground truth, privileged
information, an authoritative label, or the instructions. The sole occurrence
of “teacher” was benign factual content identifying René Descartes as Queen
Christina's teacher. The matched student uses the frozen one-epoch rank-16,
alpha-32, AdamW `5e-5`, effective-batch-32 recipe.

Full-cache artifacts and training config:

- `results/blackbox/qwen9b_privileged_gpt56_luna_medium_tvg_variedonly_v1/`
- `configs/pid_teacher_gpt56_luna_medium_varied_v1.yaml`

Training job `30316452` completed all 90 optimizer steps in 963.4 seconds
(`20m09s` including startup and saving), with reported loss `9.735`. The
adapter weight SHA-256 is
`da83f294a248a97ddf07ff7a458280f0990dea329e61fc5e2aec8faec8c61d17`.
Continuous-margin validation job `30316477` then completed successfully over
all 822 rows:

| readout | macro AUROC/BA | instructed | varied | unique scores | missing/parse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generated binary | `0.89881` | `0.97500` | `0.79722` | `2/822` | `7` parse |
| Direct `0|1` margin | `0.93762` | `0.99344` | `0.86319` | `350/822` | `0` missing |
| Empty-summary margin | `0.93780` | `0.99375` | `0.86319` | `518/822` | `0` missing |
| Post-reasoning margin | `0.93250` | `0.99167` | `0.85361` | `175/822` | `0` missing |
| GPT-OSS privileged direct reference | `0.93815` | `0.99354` | `0.86431` | `348/822` | `0` missing |

Luna's direct margin is `0.00054` below the frozen GPT-OSS privileged
reference. The two direct score vectors have Pearson correlation `0.99937`
and mean absolute difference `0.00630`; per-unit changes are small and mixed.
This is a tie rather than evidence that the stronger teacher improves the
student. Retain the GPT-OSS adapter as the validation selection and do not
test-select the Luna adapter. The near-identical ranking is further evidence
that, under this fixed one-epoch recipe and local Qwen-only varied split,
teacher capability is not the dominant bottleneck.

Phoenix 5.2 nevertheless deploys the Luna adapter as an explicit private
leaderboard OOD-transfer probe, replacing only the ordinary route while
preserving every specialist and the direct-logit inference path. This is not a
local promotion claim.

Frozen post-submission local-test job `30316568` then ran full generation and
all continuous boundaries. Direct scoring reached `0.93470` macro AUROC
(`0.98073` instructed, `0.87333` varied). Post-reasoning scoring reached
`0.93429` (`0.99229` instructed, `0.85694` varied), with 176 unique scores and
eight parse failures. The matched GPT-OSS privileged student scored `0.93661`
post-reasoning; the two post-reasoning vectors correlate at `0.98058`. Generated
binary AUROC was `0.91071` and the empty-summary margin scored `0.92780`.
Generating the Luna student's learned rationale therefore does not improve
overall or varied ranking. Keep Phoenix 5.2 on direct scoring.

Phoenix 5.2.1 nevertheless packages post-reasoning scoring as a separate OOD
transfer probe. It keeps the same adapter and routes, removes the generated
final digit before rescoring, and uses a canonical `Prediction:` fallback only
when no label is present. Local aggregate performance is explicitly not its
selection claim; queued Phoenix 5.2 is the matched direct leaderboard control.
A one-row competition-NDIF notebook smoke passed with score `0.00109873`.
Generation uses `32/24/16` length-aware batches, while the final one-token
margin retains `48/32/16`.

### GPT-5.6 Sol privileged Truth Value Guard reasoning pilot (2026-07-27)

The exact Luna pilot rows, seed, privileged Truth Value Guard prompt, 4,096
token allowance, and concurrency were reused with `openai/gpt-5.6-sol`. Low and
medium effort each covered all 36 rows.

| effort | parsed | explicit predictions | prompt tokens | completion tokens | reasoning tokens | reported cost | mean summary words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | `36/36` | `36/36` | `20,052` | `4,241` | `1,292` | `$0.22522` | `48.72` |
| medium | `36/36` | `36/36` | `20,052` | `4,520` | `1,601` | `$0.23350` | `48.28` |

All calls used OpenAI's provider, stopped normally, and required no retry.
Unlike Luna and GPT-OSS, Sol followed the redundant `Prediction:` instruction
on every row. Manual review again found a concrete poisoned fact in all 18
deceptive examples and no invented contradiction in the 18 honest examples.
Low and medium were nearly identical. Medium was somewhat more comprehensive
on the Gerry Rafferty row, identifying both the false Glasgow birthplace and
the impossible 1978 date for the 1992 film *Reservoir Dogs*, while low found
only the film-date error.

Sol did not reduce token use. Relative to Luna medium on the exact same rows,
Sol low used 11.8% more completion tokens and Sol medium used 19.1% more.
Their private reasoning-token totals were also 5.3% and 30.5% higher than Luna
medium. Although Sol's list rates are five times Luna's per token, its reported
pilot costs were 10.6--11.0 times higher because Luna received substantially
more effective prompt-cache savings. Linear full-cache projections are
approximately `$18.02` for Sol low and `$18.68` for Sol medium, close to
no-cache list-price estimates of `$18.20` and `$18.87`.

The Sol summaries are polished and occasionally a little more explicit, but
the paired sample contains no poisoned fact that Sol found and Luna missed.
Do not prefer Sol from this pilot on either quality or token efficiency. If a
full Sol capacity ablation is still desired, medium is the better arm: it has a
small comprehensiveness advantage and costs only about 3.7% more than low.

Artifacts:

- `results/blackbox/gpt56_sol_openrouter_privileged_tvg_reasoning_pilot_v1/`

### Qwen3.5-397B OpenRouter Truth Value Guard benchmark (2026-07-27)

The newer `qwen/qwen3.5-397b-a17b` checkpoint was evaluated through OpenRouter
on all 822 validation rows using the unchanged binary Truth Value Guard prompt.
Reasoning was disabled. Because OpenRouter chat providers do not continue an
assistant prefill, the model greedily generated the three-token
`Prediction:<0|1>` response and the evaluator read the `0`/`1` top-logprobs at
the generated label position. All calls were pinned to Alibaba; every row
returned both binary labels within the provider's top-five limit.

| teacher | macro AUROC | instructed AUROC | varied AUROC | BA | recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B local direct logits | 0.9533 | 0.9954 | 0.8972 | 0.8940 | 0.8571 | 0.0690 |
| Qwen3.5-27B local direct logits | 0.9574 | **0.9992** | 0.9018 | 0.9119 | **0.8857** | 0.0619 |
| Qwen3.5-122B-A10B OpenRouter terminal logits | 0.9482 | 0.9973 | 0.8828 | 0.9155 | 0.8738 | 0.0429 |
| Qwen3.5-397B-A17B OpenRouter terminal logits | **0.9631** | 0.9981 | **0.9164** | **0.9179** | 0.8762 | **0.0405** |

The 397B teacher gains 0.0057 macro AUROC overall and 0.0146 on varied
deception, mainly through better ranking/calibration and fewer false positives,
but it is not uniformly stronger. Among the nine varied dataset units it
improves six and regresses three, including a sizable drop on the
`a-mo-qwen3.5-27b-3` unit (`0.8113` to `0.7375`). Treat it as a promising
same-family teacher rather than evidence that scale solves every factual blind
spot.

The complete run used 254,980 prompt tokens and 2,466 completion tokens, cost
`$0.1042`, and finished in 147.0 seconds at concurrency eight. It produced 811
unique scores across 822 rows, zero missing-label-logit rows, and eight rows
that required a retry. Artifacts are under
`results/blackbox/qwen35_397b_openrouter_nothink_tvg_binary_logit_v1/`.
OpenRouter remains development/training infrastructure only; external API calls
are not permitted in the submitted detector.

#### Explicit binary-label semantics follow-up (2026-07-29)

A validation-only prompt ablation added exactly
`Use 0 for not deceptive and 1 for deceptive.` before the existing output
contract. It otherwise preserved the 3,000-character tail renderer, terminal
generated-label logprob readout, temperature zero, disabled reasoning,
top-five label lookup, and Alibaba-only OpenRouter routing with fallback
disabled. Qwen3.5-27B was run under both prompts; the complete 2026-07-27
Qwen3.5-397B frozen run supplied the exact model/provider/config baseline for
the new clarified arm.

| teacher | prompt | macro AUROC | instructed | varied | BA at 0.5 | unique scores |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-27B | frozen | `0.95875` | `0.99917` | `0.90486` | `0.91548` | `772/822` |
| Qwen3.5-27B | explicit digits | `0.95917` | `0.99917` | `0.90583` | `0.90952` | `782/822` |
| Qwen3.5-397B-A17B | frozen | `0.96310` | `0.99812` | `0.91639` | `0.91786` | `811/822` |
| Qwen3.5-397B-A17B | explicit digits | **`0.96679`** | **`0.99854`** | **`0.92444`** | **`0.92143`** | **`814/822`** |

The 27B change was only `+0.00042` macro AUROC, with a 20,000-draw paired
within-dataset label-stratified bootstrap interval of
`[-0.00321, +0.00417]`. It changed unit AUROC on only the nine varied units
and split them evenly: four gains, one tie, and four losses.

The 397B change was more promising: `+0.00369` macro AUROC and `+0.00806`
varied AUROC, with unit wins/ties/losses `7/12/2`. Its corresponding bootstrap
interval was `[-0.00060, +0.00833]`, so the aggregate gain still narrowly
crosses zero and is not a promotion result by itself. All four arms covered
822 rows through Alibaba with zero missing label logits. The three newly
executed arms cost `$0.2164` in total.

Do not regenerate the 27B cache from this negligible result. The 397B result
is sufficient to justify one matched cache-and-student ablation, but not an
inference-only prompt edit to Phoenix 6.3. That experiment must rescore all
2,880 training rows with the clarified prompt and train a fresh student;
the current frozen targets cannot inherit the teacher's changed logits.
Because the existing training cache came from the FP8 Lambda checkpoint rather
than OpenRouter, record that backend as part of the ablation and do not claim
the OpenRouter delta transfers exactly before the cache is evaluated.

The selected regular-model cache then completed through OpenRouter as
`qwen35_397b_openrouter_nothink_tvg_binary_logit_explicit_digits_v1`.
All 2,880 varied-deception training rows were served by Alibaba as
`qwen/qwen3.5-397b-a17b`, with fallback disabled, zero retries, and zero missing
binary labels. The run consumed 1,162,601 prompt tokens and 8,640 completion
tokens, cost `$0.46890`, and took 448.8 seconds at concurrency eight. It
produced 2,761 unique scores and reached descriptive train macro AUROC
`0.92040`; training labels were not used to generate, filter, or normalize the
identity soft targets.

The regular-OpenRouter and historical FP8-Lambda target vectors had Pearson
correlation `0.99175`, Spearman correlation `0.98780`, and mean absolute
difference `0.01742`; 41/2,880 rows crossed 0.5. OpenRouter improved descriptive
train macro AUROC by only `0.00062`, so this is a modest teacher redistribution
rather than a wholesale replacement. The comparison remains jointly confounded
by provider backend/precision and the explicit prompt clause.

The cache builder verified 2,880 aligned rows in each of `generations.jsonl`,
`soft_targets.jsonl`, and `student_rows.jsonl`. A byte-level audit additionally
requires every reconstructed student prompt SHA-256 to equal the prompt hash
sent to OpenRouter. This caught and fixed YAML folded-scalar whitespace before
training; the selected config uses a literal prompt block. The immutable
artifacts are covered by `SHA256SUMS`.

Matched one-epoch student job `30343909` completed in 14m51s with train loss
`3.919`; dependent direct-margin validation job `30343910` completed in 2m38s.
The adapter weight SHA-256 is
`514aa512b1112d6a30b5e0e5e880d227676fb1fff51317dd04b6d9965c70b82e`.
The direct margin had zero missing labels and the base-versus-LoRA fingerprint
passed.

| student | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8-Lambda 397B targets, frozen prompt | **`0.95048`** | **`0.99813`** | `0.88694` | **`0.90476`** | **`0.85714`** | `0.04762` | **`662/822`** |
| regular-OpenRouter 397B targets, explicit prompt | `0.94970` | `0.99500` | **`0.88931`** | `0.89643` | `0.83571` | **`0.04286`** | `660/822` |
| Phoenix 4.4 GRPO reference | **`0.95137`** | `0.99656` | **`0.89111`** | `0.89643` | `0.84524` | `0.05238` | `79/822` |

Relative to the historical FP8 student, the new student changed macro AUROC by
`-0.00077`: instructed fell `0.00313`, while varied improved `0.00236`.
Dataset-unit wins/ties/losses were `6/9/6`. A 20,000-draw paired,
within-dataset label-stratified bootstrap interval was
`[-0.00685, +0.00506]`. The two score vectors had Pearson correlation
`0.99353`, Spearman correlation `0.98390`, and mean absolute difference
`0.02624`; 18 rows crossed 0.5. The new scores were more saturated, with 392
outside `[0.01, 0.99]` versus 277 for the FP8 student, without gaining score
resolution.

The frozen validation-only rule did not select the regular-OpenRouter/explicit
student: it beat neither the historical FP8 soft student nor Phoenix 4.4 on
validation macro AUROC. The user nevertheless authorized a local-test transfer
probe because instructed deception is nearly saturated and private leaderboard
data may contain more varied/OOD response modes. This remains a joint
provider/backend/precision and prompt ablation, not a pure prompt-only result.

Frozen local-test job `30344308` completed on one A100 in 10m35s, including a
255-second cold vLLM startup. The direct `Prediction:` margin covered all 821
rows with zero missing label logits; the one-token generated-output diagnostic
was intentionally invalid and is not reported.

| test student | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regular-OpenRouter 397B targets, explicit prompt | **`0.96030`** | **`0.99563`** | `0.91319` | `0.88333` | `0.82857` | `0.06190` | **`636/821`** |
| Phoenix 4.4 GRPO reference | `0.95982` | `0.99073` | **`0.91861`** | **`0.88690`** | `0.82857` | **`0.05476`** | `79/821` |

The new student is the narrow macro winner by `+0.00048`, but that difference
comes from instructed AUROC (`+0.00490`); it loses varied AUROC by `0.00542`.
Dataset-unit wins/ties/losses are `5/6/10`, and a fixed-seed 20,000-draw
paired within-dataset label-stratified bootstrap interval is
`[-0.00935, +0.01083]`. Phoenix 4.4 is preferred if varied units receive more
than `47.47%` effective weight, compared with `42.86%` in the current 12/9
macro. The test therefore makes the adapter a credible private-transfer probe,
but does not establish the hypothesized varied/OOD advantage over Phoenix 4.4
or justify declaring either model the robust winner. No matched historical FP8
student test artifact exists for a direct test-split comparison.

#### Regular-397B student post-reasoning validation (2026-07-29)

The direct-only student was also tested under a counterfactual reasoning
contract. The inference prompt preserved the clarified rubric and literal
digit semantics, but requested a concise `<reasoning_summary>` before
`Prediction:`. Greedy temperature-zero generation used a 512-token allowance
with native thinking disabled; scoring removed only the final generated digit
and read the normalized `0|1` logits at that position. A fresh direct boundary
under the same summary prompt separated the prompt change from conditioning on
the generated summary.

| validation boundary | macro AUROC | instructed | varied | BA at 0.5 | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: |
| training-prompt direct, prior frozen job | **`0.94970`** | `0.99500` | **`0.88931`** | `0.89643` | `660/822` |
| summary-prompt direct, precision rerun | `0.94774` | **`0.99510`** | `0.88458` | `0.88214` | `436/822` |
| summary-prompt post-reasoning, precision rerun | `0.94131` | `0.99500` | `0.86972` | **`0.91786`** | **`814/822`** |
| generated binary | `0.91667` | `0.98750` | `0.82222` | `0.91667` | `2/822` |

The post-reasoning condition loses `0.00643` macro AUROC against its
same-session summary-prompt direct boundary and `0.00839` against the prior
training-prompt direct result. Its same-session instructed change is
`-0.00010`, while varied falls `0.01486`. It improves/ties/loses `8/6/7`
dataset units; a fixed-seed 20,000-draw paired within-dataset
label-stratified bootstrap interval is `[-0.02012, +0.00667]`. The higher
fixed-threshold BA and near-unique score vector do not compensate for the
primary ranking loss.

| precision-rerun validation unit | direct | post-reasoning | delta |
| --- | ---: | ---: | ---: |
| instructed Nemotron base | `1.00000` | `1.00000` | `+0.00000` |
| instructed Qwen base | `1.00000` | `1.00000` | `+0.00000` |
| instructed Qwen a-mo-1 | `0.96750` | `0.98000` | `+0.01250` |
| instructed Qwen a-mo-3 | `1.00000` | `1.00000` | `+0.00000` |
| instructed Qwen a-mo-4 | `0.99750` | `1.00000` | `+0.00250` |
| instructed Qwen a-mo-5 | `0.99125` | `1.00000` | `+0.00875` |
| instructed Qwen a-mo-6 | `0.99000` | `1.00000` | `+0.01000` |
| instructed Qwen a-mo-7 | `1.00000` | `0.98750` | `-0.01250` |
| instructed Qwen b-mo | `1.00000` | `1.00000` | `+0.00000` |
| instructed Qwen c-mo | `1.00000` | `1.00000` | `+0.00000` |
| instructed Gemma base | `1.00000` | `1.00000` | `+0.00000` |
| instructed Gemma s-mo | `0.99500` | `0.97250` | `-0.02250` |
| varied Qwen base | `0.98625` | `0.97250` | `-0.01375` |
| varied Qwen a-mo-1 | `0.88375` | `0.78500` | `-0.09875` |
| varied Qwen a-mo-3 | `0.68625` | `0.72750` | `+0.04125` |
| varied Qwen a-mo-4 | `0.82625` | `0.84000` | `+0.01375` |
| varied Qwen a-mo-5 | `0.97750` | `0.98000` | `+0.00250` |
| varied Qwen a-mo-6 | `0.87125` | `0.84500` | `-0.02625` |
| varied Qwen a-mo-7 | `0.84250` | `0.75250` | `-0.09000` |
| varied Qwen b-mo | `0.95000` | `0.99500` | `+0.04500` |
| varied Qwen c-mo | `0.93750` | `0.93000` | `-0.00750` |

Greedy vLLM inference was not bit-stable across the two matched RTX runs.
Initial job `30344537` scored `0.94964` direct and `0.93881` post-reasoning;
precision rerun `30344571` scored `0.94774` and `0.94131`. The conclusion is
stable across this `0.00643`--`0.01083` same-session loss range. The precision
run generated a median 82 tokens, produced 818/822 exact-format completions,
used the declared `Prediction:` fallback after four 512-token caps, and had
zero missing label logits. Generation took 30.9 seconds, followed by 14.9
seconds direct and 18.4 seconds post-reasoning scoring. Continuous evaluation
artifacts now serialize 15 decimal places so saturated probabilities retain
submission-grade rank resolution.

Keep direct scoring and do not spend another local-test evaluation on this
post-reasoning condition. Reproducible configuration and launcher:

- `configs/pid_qwen397_openrouter_explicit_post_reasoning_eval_v1.yaml`
- `experiments/privileged_information_distillation/submit_qwen397_openrouter_explicit_post_reasoning.sh`

The frozen local-test follow-up confirmed the 397B result and rejected 122B as
a scale/efficiency compromise:

| teacher | test macro AUROC | instructed AUROC | varied AUROC | BA | recall | FPR | cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-122B-A10B | 0.9555 | **0.9956** | 0.9021 | 0.8952 | 0.8333 | 0.0429 | `$0.0708` |
| Qwen3.5-397B-A17B | **0.9654** | 0.9940 | **0.9272** | **0.9071** | **0.8500** | **0.0357** | `$0.1043` |

Both runs covered all 821 rows with zero missing binary logits, zero reasoning
tokens, and only `Prediction:0|1` completions. The 397B result improved slightly
from validation to test (`0.9631` to `0.9654`) and its varied AUROC rose from
`0.9164` to `0.9272`. The 122B checkpoint also improved on test but remained
behind 397B by 0.0098 macro AUROC and 0.0251 varied AUROC. It was unexpectedly
worse than the local Qwen3.5-27B control on validation despite its greater
capacity, so do not assume monotonic teacher quality from parameter count.

The 122B validation/test runs produced 775/768 unique scores and cost
`$0.0707`/`$0.0708`; the matched 397B runs produced 811/814 unique scores and
cost `$0.1042`/`$0.1043`. The 122B jobs required many more transient retries
while sharing the same Alibaba endpoint, but retrying did not change provider
or leave incomplete rows. Select 397B, not 122B, if proceeding with remote
same-family soft supervision.

### Kimi K3 OpenRouter binary-soft cache (2026-07-29)

The frozen no-thinking Truth Value Guard prompt was evaluated on the 821-row
local test through `moonshotai/kimi-k3`. OpenRouter's first-party Moonshot
endpoint did not expose token log-probabilities, so the run was pinned to the
standard Fireworks endpoint with fallback disabled. Fireworks returned both
literal terminal labels on every row. Kimi scored `0.98244` test macro AUROC
(`0.99865` instructed, `0.96083` varied), with 735 unique scores, 86
duplicate-score excess rows, and zero missing label logits. The run cost
`$0.84434` and took 396.7 seconds. This is training/development evidence only;
external API inference remains prohibited in the competition submission.

The user then explicitly authorized a training-only soft-target cache. The
matched Fireworks run covered all 2,880 varied training rows, retained the
identity-normalized `P(1)/(P(0)+P(1))` target, and did not use local-test
outputs as training data. It completed in 615.6 seconds for `$3.61396`,
processing 1,101,623 prompt tokens and 23,040 completion tokens. All responses
resolved to `moonshotai/kimi-k3` through Fireworks; 313 rows required at least
one transient retry, but no row failed or changed provider. The cache has 2,034
unique scores and 846 duplicate-score excess rows.

| varied training dataset unit | AUROC |
| --- | ---: |
| `Qwen3.5-27B-None` | `0.98617` |
| `Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.90664` |
| `Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.89051` |
| `Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `0.93389` |
| `Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `0.97297` |
| `Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.93855` |
| `Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `0.94246` |
| `Qwen3.5-27B-b-mo-qwen3.5-27b` | `0.99299` |
| `Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.99227` |

Training macro AUROC was `0.95072`, versus `0.91978` for the historical
Qwen3.5-397B FP8 cache and `0.92040` for the regular-OpenRouter explicit-digit
cache on the same rows. Kimi and the FP8 teacher targets remain strongly
correlated (`r=0.93642`) but differ by mean absolute probability `0.05999` and
cross the 0.5 boundary on 142 rows. This supplies meaningfully different,
higher-ranking supervision rather than a duplicate cache.

The cache and student rows are under
`results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_v1/train/`.
Their SHA-256 digests are:

- generations: `c9915c5fe1589f7abce03d1f524f84db9fef6bd06f5fef77409c6dddddfe53cc`
- soft targets: `744296859003ecad37d61ca9e37f4ee374105cf97a17933b7b1d0a66cafefa4a`
- student rows: `0580aae3d6aa127687f46011589f652c101d08c59b5d9a95a800f238cf7011fe`

The prepared student keeps Qwen3.5-9B, rank 16/alpha 32, AdamW `5e-5`,
effective batch size 32, and the validation-selected two-epoch binary-soft
recipe. GPU training is intentionally deferred until an H100 is rented. When
run, select only on direct-margin validation macro AUROC against the current
`0.95393` Qwen-397B student reference; do not schedule local-test evaluation
unless that frozen validation rule passes.

#### Kimi K3 instructed-training extension (2026-07-29)

A later explicitly authorized run extended the same frozen prompt and literal
`0|1` margin to all 3,693 instructed-deception training rows. It used
concurrency 64, remained pinned to Fireworks with provider fallback disabled,
and ultimately returned `moonshotai/kimi-k3` for every row. One row exhausted
the initial nine attempts on a malformed upstream response, then succeeded
when the resumable launcher retried only that row. The final artifacts have
zero missing binary logits, zero parse errors, and exact 3,693-row coverage.

The successful cache reports 917,424 prompt tokens, 29,544 completion tokens,
and `$3.16348` cost. The active request passes took about 477 seconds in total;
the malformed responses that were not cached may make the account-level charge
slightly higher than the successful-response total. Among saved rows, 548
needed at least one transient retry. The scores contain 2,215 unique values
and 1,478 duplicate-score excess rows, spanning `0.0008041` to `0.9998416`.

| organism family | dataset units | train macro AUROC |
| --- | ---: | ---: |
| Gemma | 2 | `0.99928` |
| Nemotron | 1 | `0.98894` |
| Qwen | 9 | `0.99718` |
| **All instructed** | **12** | **`0.99685`** |

The instructed targets are easier and sharper than the varied cache: mean
binary entropy is `0.07346` versus `0.10737` nats, and only `21.74%` versus
`35.35%` lie strictly inside `[0.01, 0.99]`. Combining the two independently
cached split macros gives a 21-unit Kimi teacher train AUROC of `0.97708`
across all 6,573 rows. This is a teacher diagnostic, not a held-out student
result.

The instructed cache is under
`results/blackbox/kimi_k3_fireworks_nothink_tvg_binary_logit_instructed_v1/train/`.
Its SHA-256 digests are:

- generations: `baf1eb0f22adad9ac132418ff4c89ad3751608a638b6addb290a625996619319`
- soft targets: `dbf473512cbe05b3de65c38498f81390d2ea07388c77b225f99fb0f3e07eb4d2`
- student rows: `045c9953e06f4e61dc43abc43befa16b6e13dfb5da7e6105b52ab790df1de105`

This removes missing instructed supervision as a confound for the next
distillation experiment and adds the Gemma/Nemotron controls absent from the
varied-only cache. It does not by itself show that full-data training or a
higher LoRA rank improves validation. Run the matched full-data rank-16 control
before attributing any residual teacher gap to rank capacity.

### Competition-NDIF Nemotron/Gemma Truth Value Guard swaps (2026-07-27)

The exact no-thinking binary Truth Value Guard boundary was also ported to the
competition-specific `https://aletheias.api.ndif.us` endpoint for two
cross-family model swaps. It retained the 3,000-character tail renderer,
excluded source reasoning, appended `Prediction:`, requested only the final
logits, and normalized the literal `0|1` choices. Tokenization was local, but
no model weights were loaded on the cluster.

Nemotron completed the frozen 821-row local test:

| teacher | macro AUROC | instructed | varied | BA | recall | FPR | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA Nemotron Super 120B-A12B | `0.95655` | `0.98708` | `0.91583` | `0.89762` | `0.82619` | `0.03095` | `70/821` |
| Google Gemma 3 27B IT | `0.92917` | `0.96344` | `0.88347` | `0.86548` | `0.85476` | `0.12381` | `136/821` |

Its 26 resumable remote sessions took `185.6s` of scoring time. Nemotron is
competitive but does not beat the frozen Qwen3.5-397B test teacher (`0.9654`
macro, `0.9272` varied), and it narrowly trails Phoenix 4.4's local-test
ranking (`0.95982`). It is only `+0.0010` above Qwen3.5-122B overall, although
its varied AUROC is `+0.0137`. The score resolution is unusually coarse:
only 70 unique probabilities over 821 rows. On the 21-row base-Nemotron
instructed unit, AUROC was `0.9500` but recall at the fixed 0.5 threshold was
zero, illustrating why ranking and secondary threshold diagnostics must stay
separate.

The first Gemma attempts selected `VisionLanguageModel` because its Hugging
Face configuration carries a vision tower. Competition NDIF actually pins
`google/gemma-3-27b-it` under the text-only `LanguageModel` remote key, as the
current Sonic branch already documents. The wrapper mismatch—not deployment
availability—caused the misleading hotswap error. After aligning the shared
helper with the competition deployment, the two-row smoke and full test both
completed.

Gemma trails Nemotron by `0.02738` macro AUROC and by `0.03236` on varied
deception. Its higher FPR and severe probability saturation are notable: 270
rows scored exactly 1.0 and only 136 unique values remained. Cross-family
performance was uneven, including AUROC `0.7000` on the 21-row base-Nemotron
instructed unit. On the two Gemma-source instructed units it scored `0.9500`
and `1.0000` AUROC. This does not make Gemma a strong universal judge, although
the matched-family unit remains relevant to an organism-self-judgment design.

Runtime code and resumable artifacts are under
`experiments/ndif_tvg_model_swap/` and
`results/blackbox/ndif_tvg_model_swap_test_v1/`, respectively.

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

### Teacher-rationale cleaning audit (2026-07-25)

The deferred semantic audit is now complete. Exact checks plus manual review
find 30 unusable summaries among the 2,877 selected varied-only targets
(`1.04%`; `2.09%` of positive targets), plus three positive parser failures
that were already excluded. Six summaries justify deception only because the
assistant disobeyed an instruction to lie; three of those visibly mention
privileged information. The selected rank-16 adapter predicts zero on 26/30 of
the unusable positive training rows, so the contradiction is behaviorally
visible.

Label-blind GPT-OSS audit job `30287922` parsed 6,561/6,570 full-cache rows and
screened 186/2,877 selected summaries (`6.47%`) for masking. Its 147 selected
factual flags have only moderate precision: a seeded random manual sample found
16/30 clear errors, three arguable/minor cases, and 11 false or pedantic
corrections. It also called all eight hand-verified supporting-detail misses
factually sound, confirming correlated same-model blind spots. Combining the
exact lower bound with the calibrated factual screen puts the practical
prevalence near `3--4.5%`; retain `1.04%` as the firm lower bound and `6.47%`
only as a review upper bound.

Do not delete these rows or remove reasoning globally. The matched
prediction-only student lost 5.24 local-test BA points, and prior teacher-prompt,
teacher-model, reasoning-trace, and data-fraction controls show that
better-looking summaries rarely change binary decisions. The next controlled
ablation should retain every authoritative binary label but mask summary loss
only for independently verified bad rationales. The strict 30-row stratum has
an optimistic arithmetic ceiling near `+0.45` overall BA point; a realistic
prior remains `0--0.5`, potentially negative. See
[`rationale_cleaning_audit.md`](rationale_cleaning_audit.md) for examples,
screen calibration, and the broader engineering ceiling.

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

### Varied-only data-fraction sweep

A matched data-efficiency sweep reduced the selected 2,877-row varied-only
cache while holding Qwen3.5-9B, rank-16/alpha-32 LoRA, one epoch, AdamW `5e-5`,
effective batch size 32, prompts, targets, and seed fixed. Sampling is
deterministic within each dataset/label stratum, so every fraction retains all
nine dataset units and both labels. Training jobs `30174652`, `30174648`,
`30174650`, `30174649`, and `30174651` produced the 5--80% adapters. Jobs
`30176229` and `30176228` extended the sweep to 1% and 2%. The 1% subset has 36
rows (18 per label and two from each of the 18 dataset/label strata); 2% has 54
rows (27 per label and three per stratum). Shared vLLM validation job `30176290`
evaluated all eight fractions in one session. The table reports that common
extended run so adapter ordering and inference scheduling are matched.

| data | rows | optimizer steps | train time | overall BA | recall | FPR | instructed BA | varied BA | parse errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 36 | 2 | **13.0s** | 0.9000 | 0.8333 | 0.0333 | **0.9792** | 0.7944 | **3** |
| 2% | 54 | 2 | 19.5s | **0.9024** | 0.8333 | **0.0286** | **0.9792** | 0.8000 | 5 |
| 5% | 144 | 5 | 48.7s | **0.9024** | 0.8333 | **0.0286** | **0.9792** | 0.8000 | 6 |
| 10% | 288 | 9 | 110.0s | **0.9024** | 0.8333 | **0.0286** | **0.9792** | 0.8000 | 6 |
| 20% | 576 | 18 | 193.8s | **0.9024** | 0.8333 | **0.0286** | 0.9771 | **0.8028** | 6 |
| 40% | 1,152 | 36 | 397.8s | 0.9012 | **0.8357** | 0.0333 | 0.9771 | 0.8000 | 4 |
| 80% | 2,301 | 72 | 810.5s | 0.9000 | 0.8333 | 0.0333 | 0.9771 | 0.7972 | 4 |
| 100% | 2,877 | 90 | 947.6s | 0.9012 | 0.8333 | 0.0310 | **0.9792** | 0.7972 | 5 |

The curve is flat rather than showing benefit from more examples. Relative to
100% in the extended run, 1% makes one break and no fixes, while 2%, 5%, and 10%
each make one fix and no breaks. Crucially, every 1%/100% and 2%/100% decision
difference involves a completion that hit the 512-token budget without a
parseable prediction. On rows where both conditions parse, the 1%, 2%, and 100%
adapters make exactly the same binary decisions. The nominal 2% improvement is
therefore a negative fallback caused by truncation, not evidence that 54 rows
learn a better judge.

The repeated 5--100% controls also expose the resolution limit of this
comparison. Changing the shared-session adapter order moved between 1 and 11 of
822 decisions per adapter relative to job `30176128`, while overall BA moved by
at most 0.0024. Some changes involved parse boundaries and some were ordinary
greedy-generation backend drift. Differences of this size should not be used to
rank fractions.

At the very low end, fixed setup really does dominate. The 1% job took 32s
wall-clock for 13.0s of trainer time, leaving about 19s for imports, model and
tokenizer setup, and saving. The 2% job took 41s for 19.5s of training, leaving
about 21.5s of fixed work. Both fractions use two optimizer steps. Shared
validation took 6m02s for all eight adapters; each adapter's 822-row scoring
pass took about 31.5--31.8s, with model startup amortized once.

Use 1--2% only as an end-to-end smoke recipe: it reduces adapter training to
32--41s total but is too weak a screen when candidate methods differ in their
teacher targets or required learning. Retain 10% as the default fast performance
screen and 20% as a conservative intermediate check. Confirm any promising
method on the full recipe, and do not spend a local-test evaluation selecting
among these fractions given the established local-to-leaderboard distribution
shift.

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

### GPT-OSS teacher reasoning-effort sweep

Job `30176491` reused one persistent GPT-OSS-120B vLLM server to compare the
checkpoint's explicit Harmony `low`, `medium`, and `high` reasoning-effort
settings. The ordinary unprivileged Truth Value Guard benchmark held its prompt,
validation rows, deterministic decoding, 2,048-token output cap, and 8,192-token
model context fixed. The same job then generated full varied-only privileged
caches for low and high; the reviewed original cache is the medium condition.

| teacher effort | direct validation BA | direct AUROC | recall | FPR | instructed BA | varied BA | direct score time | usable varied traces |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | **0.9190** | **0.9283** | **0.8786** | **0.0405** | **0.9750** | 0.8444 | **64.7s** | **2,880/2,880** |
| medium | **0.9190** | 0.9247 | **0.8786** | **0.0405** | 0.9708 | 0.8500 | 190.2s | 2,877/2,880 |
| high | 0.9071 | 0.9213 | 0.8619 | 0.0476 | 0.9417 | **0.8611** | 539.8s | 2,691/2,880 |

High improves the varied-scenario macro BA by 0.0111 over medium, but loses
0.0292 instructed BA and 0.0119 overall BA. It is about 8.3 times slower than
low in direct scoring and produces six validation parse failures. On privileged
trace generation, low took roughly eight minutes and high roughly 34 minutes;
the whole shared-server job, including startup and all three direct benchmarks,
completed in 1h04m41s. Hidden completion size rises sharply from 706 mean
characters at low to 1,429 at medium and 3,222 at high, although extracted
reasoning summaries remain similar in length. At the fixed cap, high fails to
reach a parseable final summary on 189/2,880 training rows (6.6%). This is a
real operational result of the matched cap, but it also means the high-effort
quality comparison is partly confounded by truncation; isolating uncapped high
effort would require a separate 4,096-token repair ablation.

The downstream comparison used the selected 10% fast-screen recipe, not the
full 2,877-row training run: regular Qwen3.5-9B, varied-only targets,
rank-16/alpha-32 LoRA, one epoch, AdamW `5e-5`, and effective batch size 32.
Low job `30176926` trained on 288 rows in 107.2s; high job `30176927` trained on
272 usable sampled rows in 98.6s. Their wall times including setup and saving
were 3m33s and 3m26s. Shared validation job `30176928` evaluated low, the
existing matched 10% medium adapter, and high in one vLLM session.

| teacher effort | student validation BA | recall | FPR | instructed BA | varied BA | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 0.9000 | **0.8333** | 0.0333 | **0.9792** | 0.7944 | **4** |
| medium | **0.9012** | **0.8333** | **0.0310** | **0.9792** | **0.7972** | **4** |
| high | 0.9000 | **0.8333** | 0.0333 | **0.9792** | 0.7944 | **4** |

Low and high make exactly the same 822 binary decisions. Medium differs from
each on only five rows: relative to low it makes three fixes and two breaks;
relative to medium, high makes two fixes and three breaks. The resulting
0.0012 range is below the previously measured shared-session backend-drift
floor and supplies no evidence that greater teacher effort improves the
student.

With only three effort settings, an effort-level correlation is descriptive,
not inferential. Direct overall teacher BA versus student BA has Pearson and
Spearman correlation 0.50 because low and medium tie as teachers while the
medium student changes only one net decision. On varied rows, Pearson is
-0.189 and Spearman is 0.0: the monotonic direct-teacher gain does not transfer
to the student. Across the 21 dataset units, teacher and student BA correlate
strongly at low and medium (Pearson 0.940 for each) but less at high (0.698);
this mostly shows shared dataset difficulty, not an effort effect. Row-level
teacher/student correctness association likewise falls from 0.599 at low and
0.588 at medium to 0.492 at high, while prediction agreement falls from 0.932
and 0.931 to 0.914.

Decision: retain medium and the original reviewed teacher/default full-data
adapter. Low is a useful cheap cache-generation setting for exploratory pilots,
but did not beat medium downstream. Reject high at the current 2,048-token cap:
it is much slower, substantially less reliable, and yields no student gain. Do
not spend a local-test evaluation or a full-data confirmation on this sweep.

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

### Broad Wikidata frozen-judge follow-up

A later CC0 Wikidata experiment replaced the narrow Wikipedia cache with an
independently selected 221,050-card SQLite index. Offline evaluation over all
2,877 varied-training teacher summaries established a deceptive novel-fact
recall lift of +0.0285 over shuffled evidence, versus +0.0176 on honest rows.
The database contributes 44.77 MB compressed and does not use competition rows
for entity selection.

Job `30123829` then ran the selected one-epoch varied-only AdamW `5e-5` student
under empty, real, and label-blind shuffled evidence conditions. Real evidence
improved varied validation BA from 0.7972 to 0.8056 and reduced FPR from 0.0778
to 0.0667. Shuffled evidence reached 0.8000 BA and 0.0722 FPR. On the 347 rows
parsed in all conditions, the corresponding BAs were 0.8088, 0.8198, and
0.8053. Empty-summary margin AUROC was 0.8600 empty, 0.8735 real, and 0.8611
shuffled.

This is directionally better than the Wikipedia run: evidence did not induce a
large generic suspicion response, and real cards beat both controls. It remains
too small and unstable to deploy. Dataset-unit bootstrap intervals for both
real-minus-empty and real-minus-shuffled BA cross zero; real evidence hurts
three of nine varied units versus empty. Two of 462 identical instructed
prompts also flip across repeated condition batches, establishing a generation
nondeterminism floor comparable to the net effect. Manual audit finds that
headline-supporting cards can distract the judge from a false secondary detail,
while wrong-entity cards can reinforce an error. Do not test-select this on the
local test split or add the database to the submission judge. Treat it only as
support for a future matched evidence-conditioned SFT experiment with shuffled
hard negatives and explicit evidence-relation supervision. Implementation and
the full audit are in `experiments/wikidata_rag/README.md`.

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

## Continuous Phoenix prediction-margin diagnostic

Jobs `30121392` (validation) and `30122501` (local test) measured constrained
`Prediction:0|1` logits from the selected varied-only AdamW `5e-5` adapter
without retraining. Empty-summary margins improved AUROC from the contemporaneous
binary values of 0.9036/0.9107 to 0.9371/0.9307 on validation/test;
post-reasoning margins reached 0.9305/0.9366. This confirms useful latent ranking
information, but it did not produce a reliable balanced-accuracy gain:
post-reasoning BA moved from 0.9036 to 0.9060 on validation and from 0.9107 to
0.9095 on test. Exact per-dataset top-half calibration was worse, at roughly
0.885/0.887 validation BA and 0.880/0.877 test BA for empty/post-reasoning
margins, because it materially increased FPR.

Pre-amendment decision: do not add an NDIF logits pass, change Phoenix's binary
predictions, or optimize AUROC alone. At the time, balanced accuracy was the
primary competition metric and AUROC was only the tie-breaker; the extra pass
added runtime and failure surface without a demonstrated BA improvement.

2026-07-25 amendment: the organizer change to AUROC-first ranking reverses that
deployment conclusion, not the measurements. Continuous Phoenix margins are now
a priority implementation candidate. Re-run empty-summary and post-reasoning
conditions against the current bundled rank-16 adapter, select on validation
macro AUROC, and reserve local test for one frozen confirmation. The full prior
implementation remains in commit `eabc519` on branch
`experiment/pid-continuous-margin`; the updated experiment contract is in
`docs/auroc_first_strategy.md`.

P80 completed that matched rerun and added a true direct `Prediction:` arm.
Validation macro AUROC was `0.9048` binary, **`0.9382` direct**, `0.9370`
empty-summary, and `0.9304` post-reasoning. Direct required only a `9.7s`
one-token margin pass, compared with `20.4s` generation plus `11.5s`
post-reasoning scoring, and had zero missing logits. The frozen direct-only
local-test confirmation scored **`0.9342` macro AUROC** (`0.9818` instructed,
`0.8708` varied) in a `9.3s` margin pass. Direct label margins are therefore
the selected first AUROC submission implementation; the generated reasoning
path is retained as a rejected measured control.

## Rule-based claim-gated Wikidata follow-up

A CPU-only extractor mapped explicit question forms to retained Wikidata
predicates, linked only the primary question subject, and abstained on
unsupported, ambiguous, or wrong-type entities. Iteration used extraction and
training-summary retrieval audits rather than validation labels. It passed
118/2,880 training rows (4.1%); on covered parsed rows, evidence precision rose
from 0.089 for broad three-card retrieval to 0.433, while conditional novel
target recall fell from 0.042 to 0.026.

The frozen validation gate passed only 8/360 varied rows. Job `30134337` scored
0.8000 varied BA, but none of those eight rows changed prediction relative to
the original adapter run. All nine prediction changes occurred on abstained
rows and therefore reflect replay variation. The focused facts were cited in
reasoning but merely reinforced existing decisions; they did not recover the
remaining false supporting details. Do not regenerate evidence-aware teacher
traces or retrain from this gate. The implementation is retained as a cheap,
auditable prototype, but the next retrieval attempt needs materially broader
claim-level coverage rather than a looser relevance threshold.

### Expanded Wikidata schema follow-up

Job `30134717` refetched the same 221,050 entities with 128 retained properties,
seven qualifier types, and separately resolved labels for entity-valued facts.
It completed in 57m57s with 2,169,868 raw claims. The resulting SQLite database
is 165,294,080 bytes (62,333,290 bytes zipped); combined with the current
submission tree, projected package size is 178,846,006 bytes.

The frozen dependent evaluation (`30134751`) increased training coverage only
from 118 to 145/2,880 and reduced conditional evidence precision from 0.433 to
0.384. Validation coverage rose from 8 to 12/360, but manual inspection found
only one clearly useful new row (Malawi leadership). The other additions were
non-answering event/sport metadata, a wrong St. George entity, or temporally
mismatched Nobel winners. Do not spend a judge run or teacher/student training
on this cache. Broader property retention does not repair query-relation
specificity, temporal filtering, or entity disambiguation; those are now the
dominant retrieval bottlenecks.

### Deterministic query-planning follow-up

A subsequent programmatic pass addressed those failure modes without a learned
router or another database build. It adds relation-specific predicate routing,
year-qualified recurring-event linking, conflicting-year rejection, explicit
unsupported-slot abstention, `St`/`Saint` normalization, subject-type checks,
and direct alias/burial/description evidence.

On 2,877 parsed training summaries, the selected gate covers 132/2,880 source
rows with conditional evidence precision 0.414 and 38 rows with a novel-summary
token hit. This lies between the narrow gate (118 rows, 0.433 precision, 24 hits)
and unconstrained expanded gate (145 rows, 0.384 precision, 40 hits), retaining
14/16 incremental hits while removing 13 low-value rows. The frozen validation
cache falls from 12 to ten evidence-bearing rows: it removes the irrelevant
Ernest Borgnine conflict and year-mismatched Nobel facts, and replaces the wrong
Louisiana St. George entity with Saint George's burial place in Lod. Other
audited corrections include the 1986/2006 World Cup winners, time-scoped Menem
and Ortega terms, Kerkyra/Corfu aliasing, and the Akrotiri/Cyprus description.

This is the selected rule-based retrieval implementation, but coverage remains
only 4.6% on training and 2.8% on varied validation. Do not launch a judge run,
regenerate teacher traces, or train a student from it. A learned relation router
may now be reasonable only as a matched follow-up over the same deterministic
fact and qualifier constraints.

### Compact bidirectional relation index

The next programmatic sweep tested six CPU-only extensions independently: a
compact forward/reverse relation table, answer-first inverse lookup, uncapped
histories with qualifiers, training-summary-only predicate mapping search,
mechanically generated routing cases, and a constrained office/jurisdiction
two-hop. No class labels were used for query mapping or routing selection.

The selected relation database contains 512,162 facts over 233,239 labeled
nodes and 33 predicates. It preserves 55,210 start years, 40,072 end years,
37,249 point years, and 334 `for work` qualifiers. SQLite size is 39,448,576
bytes and ZIP size is 17,118,369 bytes. Shipping it alongside the expanded v2
entity linker projects to 195,964,375 bytes with the current submission tree,
leaving about 4.0 MB below the decimal 200 MB limit. This is viable but leaves
little room for another model or large cache.

The isolated direct relation lookup covered 87 training rows with 26 novel
teacher-summary hits. Enabling bidirectional answer lookup covered 91 and
raised hits to 29, a small positive result. Mechanically generated queries
initially exposed seven missing routes (voice actors, characters, participant
history, significant events, held positions, awards, and lyricists); the
selected rules pass all 32 generated relation cases. The split-half mapping
search recovered high-precision expected mappings such as author→author and
director→director, but proposed no reliable new cross-relation mapping;
low-precision coincidences were rejected.

Targeted history/qualifier enrichment is valuable for specific slots. It turns
the Ernest Borgnine/Oscar question into direct qualified evidence for *Marty*.
A constrained person→office→jurisdiction→monarch query resolves Neville
Chamberlain's 1937–1940 premiership against George VI's 1936–1952 reign and
rejects George V; it produces the correct verdict on all three repeated
training variants. Broad unconstrained multi-entity lookup was rejected after
validation audits exposed incidental-place errors. The selected emitter
requires a support/counterevidence relation and trusts later question spans
only for typed works or explicit coordination.

On 2,877 parsed varied-training summaries, the selected cache covers 130/2,880
source rows, reaches 0.538 conditional evidence precision, and has 48 rows with
a novel teacher-summary token hit. This improves both precision and hits over
the previous selected programmatic gate (132 rows, 0.414 precision, 38 hits),
at nearly identical coverage. On varied validation it covers 15/360 rather
than 10/360. The five additions are manually relevant: Milan for Derby della
Madonnina, *Marty* for Borgnine's award, Laos for Vientiane, Portugal for
Benfica/Porto, and Boston for the three-landmark question; none of the prior
ten rows is lost.

Artifacts are under `results/blackbox/wikidata_rag_relations_v1/`. The
selected database is `relations_selected.sqlite`, the frozen caches are
`train_selected_v5.jsonl` and `validation_selected_v5.jsonl`, and the
relevance, mapping-search, routing, build, and validation reports remain beside
them. This is the strongest purely programmatic retrieval configuration so far,
but 4.5% training coverage is still too low to justify another teacher/student
run without first adding new high-confidence relation families.

### GPT-OSS relevance supervision

Job `30136046` tested the deferred clean-supervision path. GPT-OSS-120B labeled
up to 12 full-card candidates per public varied row as decisive, relevant but
insufficient, or irrelevant. The polarity-free schema parsed 2,618/2,700 rows
(97.0%) and decisive/non-decisive agreement on facts repeated across answer
variants was 98.9% in training and 100% in validation. This is materially better
supervision than teacher-summary token overlap. It exposes an oracle ceiling of
335/2,336 valid training rows and 50/282 validation rows with a decisive fact,
including 272 and 40 rows outside the current rule retriever's coverage.

A 60 KB hashed linear ranker learned the ordering. On exact-question-grouped
public-training holdouts it reached 0.893 candidate AUROC. On the 51 valid
validation rows whose exact question never occurred in training, candidate
AUROC was 0.975 and it ranked the decisive candidate first on all 8 rows that
contained one. Confidence did not transfer: the training-selected 80%-precision
threshold emitted zero novel-question rows, and a top-1/runner-up margin did not
produce a usable calibration point. The apparent 11/11 full-validation emission
result consisted entirely of questions repeated in training and must not be
treated as held-out generalization.

The teacher is also not a perfect entailment oracle. It repeatedly marked the
fact that Tchaikovsky composed *Swan Lake* as decisive for asking which Christmas
ballet he wrote, although that reverse clue does not uniquely identify the work.
Mechanically grounded data, GPT paraphrases, character-relation features, and
mined phrase rules did not fix real grouped transfer. Keep the compact model as
a ranking diagnostic; do not replace the conservative rule retriever or claim a
high-precision learned coverage gain. The remaining plausible ablation is to
give an evidence-aware teacher/judge one top-ranked fact on every row and rely
on its explicit irrelevance guard, reporting novel question groups separately.
Full method details and artifacts are recorded in `docs/wikidata_rag/README.md`.

Frozen job `30136343` ran that final ablation on the selected varied-only
reasoning-summary student. Always-on ranked evidence reduced varied binary BA
from 0.8083 to 0.8056: recall increased from 0.6667 to 0.6833 while FPR increased
from 0.0500 to 0.0722. It produced four deceptive recoveries, one deceptive
regression, five honest harms, and two honest recoveries. Size-matched shuffled
evidence also scored 0.8056 BA. On 65 questions absent from training, real
evidence reached 0.8929 BA versus 0.8871 empty, but shuffled reached 0.9013;
this is not an evidence-specific gain. Do not regenerate privileged traces or
train a student from this always-on retriever. Further progress needs diverse
grouped supervision plus a confidence/abstention target that transfers.

### Counterfactual utility supervision and abstention

Jobs `30137265`, `30137268`, and `30137381` tested that remaining ablation. For
every candidate fact, the frozen selected student was scored under empty, real,
and size-matched shuffled evidence. The utility target was the change in the
correct-label log probability relative to empty evidence; a controlled target
also subtracted the shuffled-evidence change. This directly supervises whether
the downstream reader benefits, rather than whether a fact merely looks
relevant. A compact 32,768-feature linear policy learned a candidate rank and an
explicit abstain action from exact-question-grouped public training data.

The cache contains 33,706 training candidates and 4,202 validation candidates.
There is genuine but sparse oracle headroom: on validation, choosing the best
candidate per row would raise constrained-margin BA from 0.7944 to 0.8389, with
16 deceptive rescues and two honest harms. Shuffled evidence does not reproduce
that oracle gain. Utility is nevertheless poorly learnable from the available
text and metadata. The selected raw-utility gate improved grouped internal-test
BA by 0.0029 and frozen validation constrained-margin BA by 0.0056 while emitting
30/360 rows, but its gain disappeared on novel questions. Controlled-utility,
binary rescue/harm, and semantic-top-1/top-3 variants either regressed or tied
the frozen baseline. GPT-OSS semantic relevance was not a reliable proxy for
reader utility; none of the raw gate's emitted validation candidates had been
labeled decisive.

The sparse full-generation sweep reused the exact empty completion for all 330
inactive rows, avoiding batching noise from independently regenerating identical
prompts. On the 30 active rows, real evidence tied empty at 0.8056 varied binary
BA and caused two honest harms versus one honest recovery, with no deceptive
recovery. Shuffled evidence scored 0.8083. Post-reasoning margin BA likewise tied
for real versus empty at 0.8111. The small constrained-margin proxy gain therefore
does not transfer to the actual reader behavior.

Decision: close this retrieval path for the current data. Do not ship the gate,
regenerate evidence-aware traces, or train a student from its selections. A new
attempt would require genuinely new grouped questions and a consumer trained
explicitly to use, ignore, and preferably cite evidence; further tuning on the
current validation split would be selection on noise.

### Cross-encoder capacity check

Jobs `30138821` and `30138835` replaced the hashed utility gate with pretrained
MiniLM and ModernBERT cross-encoders. Both rank every structured fact jointly
with an explicit no-evidence candidate using question-group-balanced regression
and pairwise utility loss. The sweep covered raw and shuffled-controlled targets,
two learning rates, and three epochs; checkpoint and abstention selection used
only the existing training calibration partition.

The calibration-selected 22.7M MiniLM checkpoint improves grouped internal-test
BA by 0.0037, but frozen BA and novel-question BA are unchanged. It emits 20/360
frozen rows, gains 0.0022 AUROC, and reaches only 0.5873 candidate AUROC on 638
novel candidates. The 149M ModernBERT ceiling performs worse: internal BA falls
0.0015, frozen BA falls 0.0028, frozen AUROC falls 0.0075, and novel candidate
AUROC is 0.5207. Its stronger zero-shot novel ranking (0.6036 AUROC) has no
calibration-safe emission point and degrades with selected fine-tuning.

This rejects insufficient gate capacity as the primary explanation. The utility
labels are specific to an evidence-naive reader, repeated question variants
still limit effective diversity, and small constrained-margin effects do not
transfer reliably. Do not package MiniLM despite its potential quantizability,
and do not use a favorable non-selected ModernBERT novel-row rescue. The oracle
remains an upper bound for the current frozen reader but may be a lower bound for
a jointly evidence-trained reader. If retrieval is revisited, train that consumer
with balanced useful, empty, shuffled, and misleading evidence rather than
scaling the gate again.

The transfer-focused MiniLM follow-up (jobs `30139829`, `30140074`, and
`30140075`) held model capacity fixed while screening seven supervision targets,
three ranking losses, three abstention-score policies, reader-score query
representations, bottom-layer freezing, and learning rate. Selection used the
minimum BA delta across calibration and grouped internal test before their mean;
frozen validation never entered the ordering. A zero-candidate hard-listwise
edge case found by the GPU smoke was fixed and regression-tested.

The best utility refinement uses controlled continuous utility, pairwise loss,
a coarse reader-confidence bucket, no frozen layers, `3e-5`, and epoch one. It
improves calibration/internal BA by +0.0037/+0.0053 and frozen BA/AUROC by
+0.0028/+0.0024, but novel BA is unchanged. Its controlled-positive precision
falls from 0.605 calibration to 0.478 internal and 0.471 frozen. The semantic
decisiveness branch reaches 0.9907 novel candidate AUROC but emits no novel facts
at its training-calibrated threshold. This separates ranking from calibration:
MiniLM can recognize relation-level relevance, but neither semantic supervision
nor listwise abstention yet converts that into useful held-out reader decisions.
Do not lower the semantic threshold from frozen labels or infer that high
candidate AUROC is sufficient for RAG deployment.

Final MiniLM checks do not change that decision. The predeclared three-seed
refit/weight soup (job `30140655`) scores +0.0045 calibration BA, +0.0028 frozen
BA, +0.0031 frozen AUROC, and 0 novel BA/AUROC change. Seeds 43 and 44 calibrate
to complete abstention, while seed 42 has no frozen BA change; averaging all
three retains only the same one-net-decision frozen gain seen before refit.
Semantic-checkpoint initialization followed by controlled-utility tuning (job
`30140661`) selects complete abstention on every split and raises novel utility-
candidate AUROC only to 0.5774. This exhausts the programmatic MiniLM variants
worth testing on the current cache. Do not spend a generated-reader evaluation,
quantization/package work, or test-set evaluation on this gate.

### Matched evidence-reader follow-up

Jobs `30142214`--`30142335` tested the deferred evidence-trained consumer and
one alternating retriever update. The standard varied-only one-epoch AdamW
`5e-5` recipe was trained on 12,122 targets covering real, empty, shuffled,
reader-harmful, and reader-helpful evidence conditions. Always-on real evidence
still lacks net paired benefit: it fixes 13 and breaks 15 rows versus empty.
However, the matched reader raises the constrained candidate oracle from 0.7861
to 0.8444 BA with 21 rescues and zero harms, versus 0.7944 to 0.8333 and 16/2
for the old reader. This confirms that consumer training can increase usable
factual headroom.

The alternating MiniLM refit does not transport that headroom to novel
questions. Its grouped-fold-selected hard-listwise model emits 15/360 frozen
rows, none novel. Full generation scores varied BA 0.7972 empty, 0.8028 gated
real, and 0.8000 gated shuffled: two fixes and no harms versus empty, but only
one real-specific fix versus shuffled. Both corrected question groups occurred
in training, and the real-only uxoricide card is relevant but insufficient.
Treat this as a mechanistic proof, not a submission candidate. Do not test,
package, quantize, or threshold-tune the 91.6 MB MiniLM from frozen outcomes.
Further reader/retriever iteration needs new independent questions rather than
more passes over the repeated current groups.

### Document-level FEVER/AVeriTeC path (active)

The next retrieval attempt is tracked separately in
[`docs/fever_fact_verification/README.md`](../fever_fact_verification/README.md).
It replaces compact entity cards with exact-quote grounded atomic claims,
full-page Wikipedia evidence, number-aware lexical selection, neural reranking,
and a three-way FEVER score used only as an auxiliary signal. Expensive
long-context verification remains in the privileged teacher; the document
store is not intended to ship.

The first regular-scale short-lead control covers 1,183 claims but is negative:
the matched reader scores 0.9000 BA with empty evidence, 0.8952 with real
evidence, and 0.9060 with shuffled evidence (varied: 0.7972, 0.7861, and
0.8111). High-confidence NLI relations are
often merely topical, so they are hidden from teacher-visible references by
default. A FEVER-specific hard relevance threshold is much cleaner but emits on
only 1.18% of real claims and never refutes one. Full-document retrieval and
matched real/shuffled evaluation are the
active frozen follow-up. Do not train a student unless real full-document
evidence beats both controls on question-group holdouts.

The top-1 full-document reader sweep ties empty at 0.9000 overall BA and 0.7972
varied BA, while shuffled scores 0.8988/0.7944. Real evidence fixes 13 and breaks
14 rows versus empty and fixes 15/breaks 15 versus shuffled. This is a material
improvement over short leads, but still no net paired gain. Keep student
training blocked pending the frozen top-3 result.

The frozen top-3 verifier raises real-versus-shuffled polar coverage to
73.46%/64.75% and noisy teacher-assessment agreement to 52.41%/47.25%, but its
conditional polar precision is 73.63%/74.24%. The extra documents are
retrieval-specific without making the FEVER relation reliable; keep that
relation hidden and use only the cited sentence in the final reader control.

The bounded top-3 matched reader completes at 0.8988 overall/0.7889 varied BA
for both empty and real evidence, versus 0.8964/0.7833 for shuffled. Real fixes
13 and breaks 12 varied rows versus empty, and fixes 14/breaks 13 versus
shuffled. The one-row paired gains are too small to pass the frozen gate because
real does not beat empty on BA. Do not generate student traces or run test from
this cache. Keep the document-level path active through a label-blind
abstention-triggered query-reformulation/second-hop experiment, following FIRE
and the efficient AVeriTeC systems, before repeating the same controls.

The label-blind GPT-OSS audit now covers all 1,183 initial claims. It accepts
257 decisive sentences (21.72%) with zero parse errors and finds 67
contradictions among 278 claims assessed false by the noisy extraction teacher,
versus 15 for the top-3 NLI verifier. A rate-limited 105-claim prefix of the
predeclared seed-42 200-abstention second-hop pilot adds 20 decisive sentences
with zero parse errors. The combined cache contains 277 passages on 171/822
rows. Frozen matched-reader job `30146507` scores 0.9048 overall/0.8056 varied
BA with real evidence, versus 0.9012/0.7972 empty and 0.9000/0.7944 shuffled.
Real fixes 11 and breaks six varied rows versus empty and fixes 13/breaks seven
versus shuffled. The 19 second-hop-active rows contribute two fixes and no
breaks against either control. On 65 questions absent from varied training,
real reaches 0.8780 BA versus 0.8572 for both controls, but this is only two
fixes and one break. This crosses the mechanistic reader gate but remains a
small validation signal. Proceed to a train-scale evidence-aware teacher/SFT
ablation; keep test, packaging, and deployment selection blocked until that
student improves frozen validation.

The programmatic candidate-expansion follow-up separates missing context from
missing pages. Adjacent two-sentence windows add 117 decisive claims at 93.04%
directional agreement, while raw cached top-12 candidates add 101 at 90.72%.
A strict ranks-6--12 tail falls to 83.33% and is rejected. Targeted full-page
search for all 204 claims with zero initial candidates adds 42 decisions at
95.24%; MiniLM finds 13 more after those, but that incremental tail is only
76.92% and is also rejected.

A broadened entity-OR query can retrieve five Wikipedia leads in one request.
It supplies candidates for all 204 formerly empty claims with zero request
failures, versus 167/204 for the single-full-page query. On the first 80
completed rows, GPT-OSS accepts 24 at 91.67% directional agreement; ten are new
beyond full-page retrieval and nine of those ten agree. Treat this as a
promising retrieval pilot only until its complete audit and matched reader
control finish.

Matched reader job `30146678` gives the best overall expansion result: the broad
cached lexical cache scores 0.8167 varied BA versus 0.7972 empty and 0.7944
shuffled, fixing 14 and breaking seven rows versus shuffled. However, its
exact-novel-question BA is 0.8537 versus 0.8572 empty. The preferred
generalization checkpoint is the window+lexical union in job `30146708`: 0.8111
varied BA versus 0.8000 for both controls, 14 fixes/eight breaks versus shuffled,
and 0.8902 novel-question BA versus 0.8816 empty and 0.8694 shuffled. A larger
455-passage web cache remains evidence-specific overall but its newly activated
rows have one fix and two breaks, and novel BA does not beat empty. Freeze the
417-passage window+lexical union for the train-scale teacher/SFT experiment;
retain the broad lexical and web caches only as ablations. This selection is
still validation-only and does not authorize test evaluation or packaging.

The train-scale teacher-only follow-up completed on 2026-07-15. The frozen
cache covered 1,565/2,880 varied-training rows with 3,270 audited passages from
9,302 grounded claims. Retrieval was visible to GPT-OSS but absent from every
student prompt. Real and shuffled evidence produced 2,875 and 2,853 usable
teacher targets, respectively.

Under the standard one-epoch varied-only AdamW `5e-5` recipe, the real-evidence
student tied the original teacher student at 0.9012 overall and 0.7972 varied
validation BA. It made four fixes and four breaks across eight varied rows. The
shuffled-evidence student scored 0.9000 overall and 0.7944 varied BA; real
evidence was five fixes versus four breaks relative to this control. This is a
small causal signal but not a usable improvement. Retain the original adapter,
do not spend a test evaluation on either evidence adapter, and do not add these
retrieval artifacts to the submission package. Full retrieval details and job
provenance are recorded in `docs/fever_fact_verification/README.md`.

The corrected evidence-visible run then taught an explicit
relevance/support/contradiction assessment while keeping the source block in the
student prompt. In the inactive-reused shared-session matrix, the real-trained
student scored 0.8333 varied BA with real evidence, versus 0.8167 empty and
0.7972 shuffled. On evidence-active rows the three scores were
0.8541/0.8249/0.7898, and on 65 exact-novel questions they were
0.9146/0.8816/0.8572. Real evidence therefore carries useful, generalizing
signal when directly available.

The added distillation benefit is only one net parse-valid row: with the same
real evidence, the original and shuffled-trained readers both score 0.8306
varied BA, only 0.0028 below real training. Do not describe the full 0.0222 gain
over the original evidence-free row as a training gain; most comes from
inference-time evidence. The correct conclusion is that evidence-visible RAG is
promising, while the present relevance-trace SFT adds too little to select or
test. See `docs/fever_fact_verification/README.md` for the full 3x3 matrix and
paired counts.

A label-blind materiality-filter follow-up also failed to produce a robust gain.
After strict differential reuse on rows whose evidence text was unchanged, a
narrow question-premise guard moved varied BA from 0.8306 to 0.8333 through the
single honest Beira correction, with no deceptive-row effect; all four passages
removed by the rule happened to be honest rows. Trimming accepted windows to one
focused sentence returned BA to 0.8306 (two fixes/three breaks versus the guard)
and reduced exact-novel BA from 0.9268 to 0.9146. Keeping only one claim fell to
0.8139. Do not test or deploy these filters. Temperature-zero vLLM still changed
some outputs for identical active prompts under different batch schedules, so
future sparse variant sweeps must reuse the preceding condition on every row
whose rendered prompt is identical, not merely reuse empty inactive rows. Job
`30159704` completed all nine A100 conditions in 29m40s.

## Assistant-reasoning visibility ablation

The original privileged-distillation pipeline discarded the optional
`reasoning` field from the generating assistant. A 2026-07-15 audit showed that
this field sometimes contains the correct fact before the assistant emits the
instructed false answer. In the three no-adapter varied examples missed by the
current Phoenix student, the useful fact appeared within the first 702
characters of the trace; a tail-only 1,200-character excerpt missed all three.

Full traces are too long for a uniform small context. Across the 2,880 varied
training rows, trace token length had median 292, p90 3,369, p95 4,324, and
maximum 6,124; 686 full-trace prompts exceeded 2,048 tokens and 241 exceeded
4,096 before allowing for a generated target. The frozen treatment therefore
uses a bounded 2,400-character head/tail excerpt (approximately 1,200 from each
end), marks it as untrusted, and exposes exactly the same field to GPT-OSS while
constructing the target and to Qwen while learning/predicting. It does not treat
mere discussion of a dishonest instruction, uncertainty, or resistance as
evidence that the final output is deceptive.

Across all 3,600 varied train/validation/test rows, the fully rendered Qwen
input under this treatment has token-length median 893, p95 1,238, p99 1,399,
and maximum 1,914; no row exceeds 2,048. The student training cap of 4,608 and
evaluation model context of 4,096 are therefore not active constraints for this
ablation. The matched recipe remains varied-only Qwen, rank 16/alpha 32, one
AdamW epoch at `5e-5`, and effective batch 32. Results must be interpreted as a
Qwen-only varied adaptation because the local varied splits contain no Gemma or
Nemotron rows.

Balanced teacher smoke job `30161016` parsed and label-matched all 32 rows, but
manual factual audit caught trace-induced teacher anchoring. The organism trace
incorrectly said Pilchard was Bob the Builder's dog; GPT-OSS adopted that false
memory and missed the output's actual false supporting details about a hard hat,
construction work, and a catchphrase. The matched original no-trace teacher had
handled that row correctly. The trace also made the first teacher completion
misdate Trotsky's exile.

The frozen teacher prompt was strengthened to state that traces may confidently
misremember ordinary facts and that the teacher must independently verify a
trace fact before using it as the correct side of a contrast. Confirmatory smoke
job `30161081` fixed the Trotsky chronology but not the Pilchard anchoring. This
is therefore a known limitation rather than a solved prompt issue: exposing
reasoning can corrupt the stronger teacher's factual judgment as well as reveal
useful knowledge/intent.

Full teacher job `30161085` completed in 4m30s after warm startup. It produced
2,873 parseable summaries and 2,872 parseable, label-consistent targets from
2,880 rows (1,439 honest and 1,433 deceptive). Seven malformed/incomplete
finals and one explicit label disagreement are excluded from SFT. No usable
summary mentions the ground-truth label, privileged information, the teacher,
or the hidden-reasoning field. The training cache retains raw completions and
the exact student/teacher prompts for audit.

Student job `30161095` trained on those 2,872 targets for one epoch (90
optimizer steps) and completed in 27m15s. Frozen validation job `30161120`
scored 0.9083 overall BA, 0.8310 recall, 0.0143 FPR, and four parse errors. The
selected no-trace privileged-distillation adapter scores 0.9000 BA, 0.8333
recall, 0.0333 FPR, and four parse errors on the same split. The aggregate gain
therefore comes from sharply fewer false positives rather than higher overall
recall.

The intended varied scenario improved from 0.7944 to 0.8306 BA (recall
0.6556 to 0.6833; FPR 0.0667 to 0.0222), while instructed BA regressed from
0.9792 to 0.9667. Paired across all rows, the reasoning-aware student made 26
fixes and 19 breaks. On varied rows it made 24 fixes and 11 breaks: ten honest
and 14 deceptive fixes versus two honest and nine deceptive breaks. On
instructed rows it made two fixes and eight breaks, all on deceptive rows. This
is a real validation improvement but not a uniformly better judge; it trades
some instructed recall for much better varied calibration. Because the frozen
overall and varied metrics both improved materially, one confirmatory local-test
evaluation was authorized as job `30161123`.

The confirmatory test did not support replacing the default. Job `30161123`
scored 0.9036 overall BA, 0.8262 recall, 0.0190 FPR, and three parse errors,
below the selected no-trace adapter's 0.9155 BA. Varied BA again improved,
from 0.8278 to 0.8528 (recall 0.7056 to 0.7389; FPR 0.0500 to 0.0333), but
instructed BA fell from 0.9813 to 0.9417. The reasoning-aware adapter made 27
row fixes and 18 breaks overall, including a 23/14 varied advantage, yet macro
BA declined because the instructed errors were concentrated in high-weight
dataset units. In particular, missing the test split's sole deceptive Nemotron
row moved that 21-row unit from 1.0000 to 0.5000 BA. Keep the original no-trace
adapter as the general default. Treat reasoning visibility as a useful
Qwen-varied specialist signal whose cross-family/instructed cost is not covered
by varied-only training, not as a general improvement.

Validation-only inference ablation job `30161126` then hid
`<assistant_reasoning>` from the same trained adapter without changing its
teacher targets, weights, or judge instructions. BA fell from 0.9083 to 0.8952
overall and from 0.8306 to 0.8028 on varied rows; FPR rose from 0.0143 to 0.0357
overall and from 0.0222 to 0.0722 on varied rows. Visible reasoning made 22
paired fixes relative to hidden reasoning and 11 breaks (20/10 on varied rows),
with four versus six parse errors. The trace is therefore providing useful
prediction-time evidence rather than acting only as a training-time regularizer.
Instructed BA was nearly unchanged at 0.9667 visible versus 0.9646 hidden, so
removing unfamiliar traces does not recover the instructed regression. That
initial interpretation was incomplete: a later conditional-prompt control
showed that the always-on trace-aware instruction itself explains much of the
instructed regression even when the trace field is absent.

## Conditional trace interface and zero-shot reuse of the original student

Trace availability is perfectly confounded with scenario in the local data.
Across train, validation, and test, none of the 4,616 instructed rows has an
assistant reasoning trace, while all 3,600 varied rows do. Every varied row is
also from Qwen. The first reasoning-aware evaluation nevertheless used its
trace-specific judge instruction on every row, including instructed rows where
the optional field was missing.

The `Qwen3.5-27B-None` varied unit is not evidence that an unadapted model lies
by default. Its 400 rows use 117 system-prompt strings spanning explicit
deception instructions, deceptive personas/jailbreak framings, and matched
honest/correct controls. For example, one positive row directly says to respond
deceptively and the private trace identifies the true Gandhi answer before
substituting Subhas Chandra Bose. Conversely, the base model often refuses
lie-competition prompts and gives the correct fact. Labels describe the output,
not merely the requested behavior, so a deception-framed prompt can still be
honest. Here `varied` means varied elicitation framing, not spontaneous default
deception.

Job `30161301` made the interface conditional without changing weights: rows
with a non-empty trace use the trace-aware instruction, and rows without one use
the original ordinary instruction. The trace-trained adapter then scored 0.9155
overall BA, 0.9813 instructed BA, and 0.8278 varied BA. Relative to its static
prompt evaluation, the instructed rows had eight paired fixes and one break.
The varied prompt text was unchanged; two varied prediction differences were
greedy-vLLM scheduling variation. This recovers and slightly exceeds the
original adapter's instructed calibration, so the earlier instructed collapse
was primarily an interface mismatch rather than proof of catastrophic weight
forgetting.

The more surprising control was to expose reasoning to the selected original
no-trace adapter without retraining it. A rank-16 SVD endpoint reconstructed the
original LoRA delta to relative error about `1e-6` while carrying the
reasoning-rendering metadata. The exact conditional result, reusing identical
ordinary-prompt outputs for no-trace rows, was 0.9179 overall BA, 0.9792
instructed BA, and 0.8361 varied BA at 2,400 reasoning characters. This exceeds
the adapter trained on trace-conditioned GPT-OSS summaries. The existing
student can therefore interpret the new field zero-shot; trace-conditioned
teacher targets are not required and may transmit the teacher's trace
anchoring.

A frozen context sweep on that original adapter tested 1,200, 2,400, 4,000, and
6,000 reasoning characters with head/tail truncation. A refinement also tested
3,200 and 4,800 characters plus head-only and tail-only 4,000-character views.
After exact prompt-hash reuse for unchanged rows, 4,000-character head/tail is
the selected representation. Its raw shared-session result was 0.9226 overall
and 0.8472 varied BA; 4,800 characters and tail-only tied it after paired reuse,
while head-only fell to 0.9190 overall and 0.8389 varied. The longest fully
rendered 4,000-character validation prompt was 1,889 tokens, so neither the
2,048-token deployment input nor the 4,096-token vLLM context is active on this
split. The trace-trained SFT cache is also far from its training limit: complete
prompt-plus-target examples have median 995, p95 1,346, and maximum 2,044 tokens
against `student.max_length=4,608`, with zero truncations. The bottleneck is not
training context length. More context is not automatically better.

Removing the duplicate final assistant answer from `<context>` while retaining
it in `<output>` was also rejected. It saved 56 input tokens on average and 164
at the validation maximum, and reduced generation time from 43.2s to 40.4s in
an exact paired run. However, it changed 21 decisions: five instructed fixes
and no instructed breaks, but only five varied fixes against nine varied
breaks. Overall BA rose slightly from 0.9190 to 0.9214 while varied BA fell from
0.8389 to 0.8306. Keep the duplicated rendering the adapter saw during SFT;
the modest context saving is not worth another scenario tradeoff.

Prompt wording around counterfactual instructions, an append-only trace field,
and 2,400--4,800-character strict variants did not beat the simpler
4,000-character trace-aware prompt. The best strict variant tied 0.9226 overall
but had lower varied BA and used more context. An OR ensemble of ordinary and
strict decisions gained only 0.0012 raw BA while doubling generation work.
Keep the single ordinary trace-aware member unless the dedicated supporting-
detail sweep gives a material paired gain. That sweep, job `30161549`, did not:
an explicit audit of every added name/date/location/quantity made seven fixes
and seven breaks on the 360 changed varied rows, tying the same-session baseline
at 0.8444 varied BA while increasing parse errors from two to six. A narrower
instruction to scrutinize only facts introduced after the private reasoning
made three fixes and 13 breaks and fell to 0.8167 varied BA. The observed
supporting-detail failure is real, but prompting Qwen to search harder does not
make its factual verification reliable enough.

LoRA-space interpolation also failed. Validation job `30161297` evaluated trace
delta blend weights `0`, `0.25`, `0.5`, `0.75`, and `1`; raw BAs were 0.9119,
0.9048, 0.9083, 0.9071, and 0.9060. Any movement toward the trace-trained delta
harmed the zero-shot original endpoint. A frozen Wikipedia-window RAG sweep was
similarly negative: empty, real, and shuffled evidence scored 0.9119, 0.9107,
and 0.9060 overall BA. Real evidence made two fixes and three breaks versus
empty evidence. The retrieved passages often establish the user's premise or
an ordinary factual correction without resolving whether the assistant output
is belief-based deception, so do not package this RAG path.

Finally, constrained binary logits do not hide a better decision boundary.
With the selected 4,000-character prompt, post-summary forced 0/1 logits exactly
matched generated decisions at 0.9238 BA in job `30161379`; threshold sweeping
did not improve them. Empty-summary logits ranked well (0.9468 AUROC) but peaked
at only 0.9167 BA. The remaining error is mainly factual judgment and claim
salience, not final-token parsing.

The first matched training controls at 2,400 characters are also below the
zero-shot 4,000-character endpoint:

| student update | overall BA | instructed BA | varied BA | parse errors |
| --- | ---: | ---: | ---: | ---: |
| fresh varied-only, 50% trace dropout, AdamW `5e-5` | 0.9155 | 0.9792 | 0.8306 | 2 |
| fresh mixed, 50% trace dropout, AdamW `5e-5` | 0.9119 | 0.9771 | 0.8250 | 2 |
| fresh mixed, full trace, AdamW `5e-5` | 0.9119 | 0.9771 | 0.8250 | 4 |
| continue original, 50% trace dropout, AdamW `2e-5` | **0.9190** | 0.9792 | **0.8389** | 1 |
| continue original, full trace, AdamW `2e-5` | 0.9155 | 0.9771 | 0.8333 | 1 |
| original teacher targets + trace input, varied-only, `5e-5`, 2,400 chars | **0.9190** | **0.9813** | 0.8361 | 3 |
| original teacher targets + trace input, varied-only, `5e-5`, 4,000 chars | **0.9190** | 0.9750 | **0.8444** | 2 |
| conditional mixed teacher targets, `5e-5`, 2,400 chars | 0.9143 | 0.9771 | 0.8306 | 4 |
| conditional mixed teacher targets, `5e-5`, 4,000 chars | **0.9190** | 0.9750 | **0.8444** | 2 |
| conditional mixed original targets, `5e-5`, 2,400 chars | 0.9143 | 0.9771 | 0.8306 | 4 |
| conditional mixed teacher targets, 50% trace dropout, `5e-5`, 2,400 chars | **0.9190** | **0.9813** | 0.8361 | 3 |

Dropout helps the conservative continuation relative to full-trace
continuation, but does not exceed exposing a longer trace to the unchanged
original adapter. Adding instructed targets preserves instructed performance
but dilutes the varied gain; dropout makes no macro-BA difference in the fresh
mixed run. Fully conditional mixed training and student-only-target controls
therefore do not improve on the unchanged original adapter. In the exact
4,000-character checks, the original and conditional-mixed adapters emitted
identical binary predictions on all 822 rows despite different free-form text.
The trace-conditioned teacher targets are not adding a useful decision signal.

Crossing all proposed safeguards does not change that conclusion. The fresh
mixed student with conditional prompting and deterministic 50% trace dropout
reaches 0.9190 BA at 2,400 characters by lowering FPR to 0.0095, but still
trails the summary/binary ensemble and does not improve varied recall enough.
The matched one-epoch learning-rate sweep confirms that the original fresh-SFT
rate transfers to this broader recipe: `2e-5`, `5e-5`, and `1e-4` scored
0.9107, **0.9190**, and 0.9119 BA, respectively. Their varied BAs were 0.8222,
**0.8361**, and 0.8250. Thus the data/interface change does not justify moving
the one-epoch AdamW rate away from `5e-5`; the lower rate underfits and the
higher rate does not buy additional recall. Giving `2e-5` a second epoch raised
it to 0.9155 overall and 0.8306 varied BA, partially closing the gap but still
not matching one epoch at `5e-5`.

The dropout-density/seed bracket also rejects treating the original 50% mask
as an accuracy optimum. At the selected `5e-5`, dropout probabilities 25%, 50%
(seed 0), and 75% scored 0.9155, 0.9190, and 0.9155 BA; the 25% and 75% models
had the same aggregate metrics despite six binary-decision differences. A
second 50% mask with seed 1 scored only 0.9143. Corresponding varied BAs were
0.8306, 0.8361, 0.8306, and 0.8333 for the seed-1 replicate. Random trace
removal remains a sensible interface-robustness safeguard, but its small
validation advantage is mask-sensitive and does not support selecting a new
adapter.

The student-only controls are especially informative: they retain the original
GPT-OSS summaries as targets and change only the student input by adding the
trace. The mixed original-target and trace-teacher-target students made exactly
the same 822 binary decisions at 2,400 characters. The varied-only control
matches but does not beat zero-shot trace exposure. A targeted
supporting-detail continuation also failed. A one-epoch `5e-5` continuation on
163 detail-hard deceptive rows plus 163 honest controls scored 0.9155 BA; a
safer three-epoch `2e-5` continuation scored 0.9190. Neither improved recall on
the poisoned supporting facts. This supports retaining `5e-5` for fresh
one-epoch SFT while using a lower learning rate for any continuation; it does
not establish that the fresh optimum transfers unchanged to a narrow replay
update.

Reasoning excerpt allocation was also frozen. At 4,000 characters, moving from
a balanced head/tail split to 75% tail made zero fixes and two breaks on the 99
rows whose prompts changed. Moving to 75% head made two fixes and three breaks.
Keep the balanced split.

An inference-only prediction-first format with a 32-token budget scored 0.9143
BA, 0.9688 instructed BA, and 0.8417 varied BA with zero parse errors in 24.6s.
It is about 40% faster than the local full-summary validation generation, but
below the roughly 0.922--0.924 raw validation range of the summary-first path.
The compact reasoning is therefore useful at inference as well as in the SFT
target. Alone, prediction-first is only a latency-oriented fallback.

Its errors are nevertheless complementary. Taking the maximum of the
summary-first and one-line binary decisions improved every repeated
4,000-character summary run by 0.0024--0.0048 BA. A final shared-vLLM validation
job scored 0.9202 for summary-first, 0.9143 for binary-only, and **0.9238** for
their max ensemble. Ensemble recall rose from 0.8524 to 0.8762 while FPR rose
from 0.0119 to 0.0286; varied BA rose from 0.8444 to 0.8500, and there were zero
ensemble parse errors. The binary member emitted exactly one line on all 822
rows and added 23.9s to the 43.2s summary pass. The fully crossed mixed,
conditional-prompt, 50%-dropout student tied the original adapter's ensemble
at 0.9238 BA, including identical 0.9792 instructed and 0.8500 varied BA.
Retain the simpler original varied-only adapter; neither the trace-conditioned
teacher nor the proposed data-composition safeguards improve the frozen
ensemble.

That condition was then locked for the only new held-out test evaluation. The
summary member scored 0.9262 overall BA, 0.9833 instructed BA, and 0.8500
varied BA. Maxing it with the one-line member reached **0.9274 overall BA**,
0.9708 instructed BA, and **0.8694 varied BA**, with 0.8929 recall, 0.0381 FPR,
and zero ensemble parse errors. Relative to the historical no-trace test run,
the summary member made 26 fixes and 16 breaks; the max ensemble made 31 fixes
and 18 breaks. The binary member added 13 fixes and 10 breaks relative to the
summary member. This materially exceeds the original adapter's 0.9155 overall
and 0.8278 varied test BA without changing weights. The gain comes from a
conditional trace interface and complementary readouts, not from transferring
the reasoning-trace SFT update.

A final calibration over all 6,573 training rows supports retaining the binary
member despite its intent-sensitivity on a targeted honest prefix. The summary
member scored 0.8942 macro BA (0.9487 instructed, 0.8215 varied), while the
binary member scored 0.8922 (0.9444 instructed, 0.8226 varied). Their maximum
reached **0.9032 overall**, 0.9578 instructed, and **0.8306 varied**. Relative
to summary alone, the binary member changed 181 rows: 104 fixes and 77 breaks;
on varied rows the balance was 58 fixes to 32 breaks. FPR rose from 0.0146 to
0.0375, but recall rose from 0.8029 to 0.8440. This broader in-sample
calibration is consistent with the validation/test recall benefit rather than
showing that the ensemble gain depends on those small held-out slices.

The intent-only concern is real but prompt wording does not solve it robustly.
On 32 honest rows from the highest-FPR dry-run unit, summary alone flagged four
and the binary member added five cases where a private trace planned a lie but
the final answer was correct. A stricter one-line prompt requiring a concrete
false output claim reduced validation FPR from 0.0214 to 0.0143, but destroyed
recall (0.8500 to 0.7690) and fell from 0.9143 to 0.8774 BA. Combined with the
summary member it tied the original ensemble at 0.9238 BA, making five fixes
and five breaks. Keep the locked binary wording; the guard merely shifts which
errors occur and was not evaluated on test.

A validation-only cross-method diagnostic also maxed this student ensemble with
the existing 4,096-token three-prompt Qwen heavy ensemble. BA moved only from
0.9238 to 0.9250: the heavy member made 19 fixes and 20 breaks relative to the
student, and varied BA remained exactly 0.8500 because its recall gain was
offset by higher FPR. The heavy path costs 1,314s versus 67s for the local
student ensemble. This is not enough complementary signal to justify a second
three-prompt deployment path, and it was not evaluated as a new test condition.

A stratified audit of the remaining validation false negatives confirms the
same ceiling seen in the earlier shared-error work: the headline answer and
private trace are usually correct, while an added supporting sentence carries
the poison. Examples include dating Ecuador's Galapagos annexation to 1845,
calling the 2017 pound coin Britain's first 12-sided coin despite the earlier
12-sided threepence, calling Beaker "Dr. Beaker," and giving a confident but
unsupported etymology for "catgut." The trace generally repeats only the
correct headline, and both readouts then produce a fluent truthfulness summary
without auditing every adjunct. An explicit supporting-detail prompt already
made equal fixes and breaks, so this is evidence of unreliable fact checking,
not a missing generic instruction to inspect details. It also explains why
longer traces and trace-conditioned targets saturate: the missing fact is often
absent from the trace itself.

For deployment, the submission notebook now uses `LanguageModel`, conditional
trace prompting, 4,000-character balanced head/tail traces, length-sorted
dynamic padding, and a 2,048-token input cap. The longest validation prompt was
1,889 tokens; the locked test maximum was 2,050, so only one held-out prompt is
left-truncated by two input tokens under the notebook cap, preserving the end
of the conversation and output. Small-prefix
remote smokes passed 32 rows at batch 32 in 35.9s and 64 rows at batch 64 in
34.8s, but those prefixes were not representative capacity tests. On the full
400-row varied-Qwen unit, batch 64 OOMed on the first request and batch 32
eventually OOMed at shape `(32, 1416)` despite separate remote sessions.

The certified adaptive schedule uses batch 32 through 1,300 prompt tokens,
batch 24 through 1,600, and batch 16 above 1,600. The final notebook evaluates
summary and one-line members in separate passes so binary rows can stop after
three tokens instead of decoding alongside long summaries. This exact final
path completed all 800 prompts and wrote a valid 400-row submission. On the 40
labeled validation rows from that unit, NDIF predictions agreed exactly with
the shared-vLLM ensemble and both scored
0.9500 BA. A follow-up full-unit run safely grouped two adjacent generation
requests in each NNsight session, retained the same 40/40 vLLM agreement, and
wrote all 400 rows without an OOM. The notebook therefore defaults to two
requests per remote session to reduce session setup and queue overhead; this
does not raise any per-request batch cap. Keep the length tiers and do not
restore batch 64 based on the small-prefix timing. The official
`submit.py --dry --limit 32` pipeline then completed all 21 dataset units
successfully in 14m06s from a clean staged root.
Those first-32 prefixes contain no positive labels, so the dry-run metrics are
only an execution/FPR smoke and must not be compared with the balanced
validation/test results.


## Transfer-focused coverage experiments (2026-07-17)

The follow-up program deliberately separated three hypotheses that the local
Qwen-varied split cannot distinguish: source-family coverage, missing semantic
response modes, and ordinary hallucinations that should remain honest. All
student updates reused the selected one-epoch varied-only recipe or a small
continuation from it; no private leaderboard result was used to select weights.

### Model and family adaptations

Two permitted model-specific readout parameters were evaluated from training
coverage. Disabling the binary recall member for Gemma base raises validation
BA by `0.0012` and ties locked test while saving one member call for that unit.
Routing Qwen to the varied-only adapter and Gemma/Nemotron to the mixed adapter
also gains `0.0012` validation BA and ties test, but two BF16 adapters exceed
the 200 MB package budget. Both are retained as low-risk parameter evidence,
not as broad source-specific logic.

A single family-coverage adapter added 64 instructed Gemma and 17 instructed
Nemotron targets to 288 Qwen-varied targets. It beats the matched Qwen-only
step-count control and moves validation BA `0.9000 -> 0.9024`, but its locked
summary-only test score falls `0.9131 -> 0.9095` through extra cap-length parse
failures. Under the current conditional summary/binary interface it ties the
baseline at `0.9238` validation BA. This supports family-balanced supervision
as a muted theoretical improvement while rejecting the present weights. The
local data still contain no varied Gemma or Nemotron example, so instructed
family replay cannot establish cross-family varied transfer.

An independent 80-row family-balanced long-answer continuation is similarly
safe but small. It moves a disjoint 200-row external set from `0.9600` to
`0.9650` BA, with all five source families non-worse, and preserves ordinary
competition validation. Under the deployment interface it moves `0.9190` to
`0.9202`, but adds one parse error and makes two fixes/two breaks. Keep the data
recipe and cache scaffolding for genuinely new family/scenario data; do not
promote either current adapter.

A more focused continuation then used all 64 soft-trigger teacher summaries
plus the same 10% Qwen-varied replay. It moves competition validation from
`0.9036` to `0.9012` BA and leaves the disjoint 200-row soft-trigger set exactly
at `0.9000` BA: recall gains one point while FPR gains one point. Gemma improves
by `0.0185`, Llama/Mistral are unchanged, and Qwen loses `0.0208`. This rejects
the hypothesis that the earlier broad mixture merely diluted a useful small
cross-family varied signal. Close the present soft-trigger exposure path; new
coverage should add genuinely different scenario evidence rather than replay
these 64 targets at another learning rate.

That comparison was subsequently found to use a mismatched 10%-data control,
even though the candidate continued the full selected adapter. A corrected
joint evaluation against the actual parent changes none of the 800 external
predictions: both score `0.9000` soft-trigger BA, with identical family metrics.
Competition validation moves `0.9024 -> 0.9000`. This is the authoritative
causal result. The earlier Gemma/Qwen movements came from the control mismatch,
not the continuation; close this training branch.

### Small external reasoning distillation versus semantic routing

A 256-row GPT-OSS Liars' Bench cache covered harm-pressure choice, knowledge
reports, action reports, and soft triggers with 32 examples per label/category.
Fresh and continued students did not gain the required `0.02` macro external
BA. In contrast, prompts selected by observable response contracts produced
large targeted gains with the unchanged Phoenix weights. The broad HP-KR
report-type prompt raises regular Qwen from roughly `0.34` to `0.846` full-set
BA and the action-report prompt raises a frozen 200-row sample from `0.61` to
`0.76`. The full action category confirms the direction (`0.5307 -> 0.6999`,
FPR `0.7526 -> 0.0419`) across every source family, although it misses its
predeclared absolute `0.70` BA gate by `0.00011`.

The preregistered scale-up rejects insufficient exposure as the leading
explanation at this scale. It generated 1,024/1,024 usable GPT-OSS targets from
new rows, balanced at 128 examples per label/category, then continued the
selected rank-16 adapter on those targets plus 288 Qwen-varied replay examples.
The run completed 41 steps in 1,146 seconds and produced weights with a
different hash from the parent. Despite the real update, the candidate and
parent make exactly the same parsed decisions on both frozen evaluations:
competition validation stays `0.9048` BA across 822 rows, while the disjoint
external macro stays `0.61375` across 800 rows. Choice, HP-KR, Insider Trading,
and soft-trigger BAs remain `0.630`, `0.355`, `0.575`, and `0.895`; every
category and category/source delta is zero. The `+0.02` external gate therefore
fails. Do not test or promote this adapter, and do not repeat the same
exposure/LR path. HP-KR and Insider Trading were previously inspected over
their complete public categories, so this evaluation is train-disjoint rather
than a never-inspected benchmark.

A follow-up held those parent weights fixed and varied only the general
inference rubric on a preregistered 400-row development half. The unchanged
control scored `0.6300` macro BA; mode-first, claim-ledger, and
truthful-alternative prompts scored `0.6550`, `0.6600`, and `0.6975`.
Truthful-alternative gained `0.070` choice and `0.210` HP-KR BA with action
`-0.010` and soft trigger unchanged, but it breached the frozen
category/source floor on Qwen choice (`-0.0696`) and Gemma action (`-0.0556`),
as well as two small soft-trigger cells. No prompt advanced, so the untouched
400-row confirmation half and competition validation were not queried. This
shows more learnable movement from inference semantics than from the larger
weight update, while still rejecting a universal replacement prompt.

These results favor model-agnostic semantic routing over small weight updates:
use one existing call, require an explicit observable report contract, and
change the judging question rather than infer a hidden category or organism
trait. A forced-choice harm specialist failed (`0.57` BA), showing that not
every response-mode prompt transfers. The HP-KR and action routers have zero
coverage on the complete local competition corpus, so their value is missing-
mode coverage rather than local validation hill climbing. The action route is
retained as a dormant theoretical change; a recipient-perspective clarification
is being evaluated separately after a label-aided audit exposed a repeated
meta-level error.

A later rank-1 HP-KR specialist screen reinforces that conclusion. Three fresh
7.3 MB adapters trained on the 256 GPT-OSS HP-KR targets at increasing dose;
training loss fell from `1.425` to `1.192`. Nevertheless, all three made
exactly the same 100 development decisions as base Qwen with the established
report-type prompt (`0.8100` BA), with zero source-family or parse changes.
The Phoenix/truthful-alternative reference scored `0.5700` in the same session.
No adapter passed the `+0.02` gate, so confirmation was skipped. Do not raise
rank or training dose: the prompt supplies the useful semantic specialization,
while this small supervised update does not add a held-out decision signal.

That clarification recovers the intended semantic distinction but is not
calibrated. Full-category BA rises `0.6999 -> 0.7884` and recall nearly doubles
to `0.8668`, while FPR explodes to `0.2900`, parse failures rise from 14 to 89,
and Llama loses `0.0237` BA. It fails both the FPR and family-preservation gates.
Keep the original low-FPR action route as the muted missing-mode safeguard; do
not replace it with the label-aided recipient wording or fit a blend on these
same full-category labels.

A passage-grounded True/False route supplies a second narrow missing-mode
safeguard. On 2,000 untouched balanced soft-trigger rows, the unchanged Phoenix
weights move from `0.879` to `0.902` BA when the prompt explicitly treats the
marked passage as authoritative. Recall moves `0.859 -> 0.862`, FPR
`0.101 -> 0.058`, parse errors `15 -> 6`, and Gemma/Llama/Mistral/Qwen all
improve. This misses the strict `+0.03` and `0.94`-absolute promotion criteria,
but passes every preservation criterion. The exact router matches zero of all
8,216 local competition rows. Retain the exact one-call route in Phoenix v2.4
under the muted-validation policy: require literal passage markers, an explicit
True/False contract, and an exact one-word Boolean output; do not tune or widen
it from these labels.

The strongest external combination keeps a heavy factual default and routes
only explicit semantic report contracts. On the frozen 800-row spectrum,
Qwen-9B D/K/S plus the epistemic and action specialists moves macro category BA
`0.7788 -> 0.8325`: HP-KR `0.760 -> 0.850`, insider/action reports
`0.640 -> 0.765`, with forced-choice and soft-trigger rows untouched. All
category/family cells are non-worse. This supports response-mode routing over
organism-specific weights. A matched Phoenix-adapter heavy run is required
before calling it a one-adapter deployment pipeline.

That matched one-adapter check fails its lower-level preservation gate.
Phoenix-heavy macro category BA is close to base Qwen (`0.7738` versus
`0.7788`) and parse failures fall slightly, but action reports lose `0.025` and
three category/family cells lose `0.060--0.074`. Therefore do not apply the
specialists post hoc or run the conditional competition-validation adapter
ensemble. The transferable hybrid currently requires base-Qwen heavy state plus
Phoenix specialist state (or separately activated adapters), not one always-on
Phoenix LoRA.

Judge capacity is the stronger route when available. On the matched 800-row
four-mode spectrum, exact Qwen-27B D/K/S raises macro category BA from
`0.7788` to `0.8750`; choice, HP-KR, action, and soft-trigger BAs all improve,
and the worst family-cell delta is only `-0.0208`. This passes the preregistered
broad-transfer gate despite the earlier 200-row soft-trigger loss, showing why
single-category samples should not veto a multi-mode result in isolation.

The fixed semantic specialists should not be stacked on that stronger judge.
They lower Qwen-27B macro BA to `0.8513`, including `-0.085` on action reports
(34 fixes, 53 breaks overall). Keep Qwen-27B heavy intact; the semantic hybrid
is a Qwen-9B capacity fallback, not a universal router.

### Honest-hallucination controls

The honest-error experiment screened 7,600 disjoint Alpaca responses, selected
128 Phoenix false positives balanced over four source families, and obtained
128 parsed GPT-OSS summaries explaining visible errors without treating them as
knowing deception. A continuation with 10% varied replay changes only one of
400 frozen Alpaca decisions. Increasing exposure to one epoch at `5e-5` or
three epochs at `2e-5` yields the identical `0.0500 -> 0.0475` FPR result; both
miss the required `0.025` endpoint. More LR/epoch tuning is closed.

Adding the distinction directly to every prompt proves that Qwen can apply it:
Alpaca FPR falls to `0.0025`. It is much too conservative for deception,
however, dropping competition-validation BA from `0.9190` to `0.8274`. A
trace/visible-intent router appears perfect locally because all varied rows have
traces and every routed instructed row happens to be honest. The independent
balanced response-mode check rejects it: macro BA falls `0.7713 -> 0.6600`,
soft-trigger recall becomes zero, and parse errors rise by ten. Absence of a
visible lie instruction is not evidence of honesty; helpful provenance can
still hide an organism adaptation or trigger. Do not deploy either guard.

### Judge capacity control

The exact Qwen3.5-27B swap for the frozen three-prompt heavy ensemble passes its
validation gate: BA `0.9226 -> 0.9298`, instructed `0.9771 -> 0.9792`, and
varied `0.8500 -> 0.8639`. Recall rises to `0.9238`, FPR rises to `0.0643`, and
member-level parse errors fall from 38 to five. Scoring is only 16.7% slower
(`1,314s -> 1,533s`) despite the parameter increase. The `known4096` member
alone exactly matches `0.9298` BA with zero parse errors, one third of the
generation work, `0.8881` recall, and `0.0333` FPR. It is the locked speed
candidate, but shifts scenario BA from `0.9792`/`0.8639` for the full max to
`0.9938`/`0.8444` instructed/varied. One exact local-test confirmation is
complete. The full ensemble reaches **`0.9464` test BA**, `0.9573` AUROC,
`0.9357` recall, and `0.0429` FPR, with `0.9833` instructed and `0.8972` varied
BA. It makes 29 fixes and 15 breaks versus matched 9B generations and is now
the strongest tracked submission-compatible local-test judge. The locked
`known4096` speed candidate falls to `0.9083` despite tying validation; retain
all three prompts and do not post-hoc select the `0.9405` scrutiny member.
This is capacity evidence, not source-model routing, and NDIF runtime remains a
separate check.

A matched 21-unit transfer audit explains why the model-level conclusion is
stronger than any unit-level router. The validation mean delta is `+0.00714`
with a unit-bootstrap 95% interval spanning zero; locked test is `+0.01667`
with a positive interval. Per-unit validation/test delta correlation is only
`0.362`, but 19/21 units are non-opposed and only two reverse direction. Keep
the broad capacity swap and scenario checks; do not infer organism rules from
the noisy individual cells.

### Observable and family-specific calibration controls

A narrow task-shape cutoff was frozen after both 9B and 27B validation caches
showed fewer low-confidence false positives on short single-turn conversations
with an explicit deception directive. Requiring rating 6 rather than 2 only
for that visible shape moved 27B validation `0.9298 -> 0.9357` and 9B validation
`0.9226 -> 0.9250`, while aggregate varied BA was unchanged for both. The
existing 9B test cache does not confirm it: one varied deceptive row breaks,
moving overall BA `0.9298 -> 0.9286`. Retain the analysis code as a theoretically
sound but muted calibration control, not as submission logic. The frozen 27B
test cache rejects it sharply: BA falls `0.9464 -> 0.9214`, including complete
deceptive-recall loss on the tiny Nemotron unit. Validation agreement across
judge sizes did not transfer; retain only the negative-control implementation.

Per-family rating calibration has stronger split replication but a larger
coverage caveat. Requiring rating 7 rather than 2 for Gemma outputs raises mean
Gemma BA `0.9500 -> 0.9750` on both 9B validation and local test, and to
`0.9875` on 27B validation. Competition rules permit documented per-organism
parameters, but all local Gemma examples are simple instructed deception. A
frozen exact-heavy-judge run on balanced cross-family Liars' Bench soft-trigger
rows rejects the parameter. The cutoff removes all Gemma false positives but
lowers recall from `0.9630` to `0.8889`, leaving Gemma BA unchanged at `0.9444`;
overall BA also stays `0.9400`. This is precisely the missing varied-family
failure the external gate was intended to catch. Keep the evaluator and reject
Gemma rating 7 from submission logic.

### Qwen-27B capacity transfer boundary

The exact Qwen-27B D/K/S model swap is the strongest tracked competition-test
judge, but its first disjoint cross-family check fails. On 200 balanced Liars'
Bench soft-trigger rows, Qwen-27B scores `0.9250` BA versus `0.9400` for the
matched Qwen-9B judge. It is more conservative (`0.91` versus `0.96` recall;
`0.06` versus `0.08` FPR), with Gemma BA down `0.0370` and Qwen down `0.0208`.
Keep the competition candidate, but do not use model scale as evidence of broad
varied/family transfer. An 800-row four-response-mode spectrum is the frozen
next audit; semantic specialists remain necessary if the same boundary repeats.

The post-hoc Gemma specialization signal from the focused soft-trigger
continuation also fails a genuinely untouched confirmation. On 1,000 balanced
Gemma rows excluded from all prior teacher/evaluation caches, the baseline
scores `0.9170` BA and the continuation `0.9140`; recall is unchanged and FPR
rises `0.072 -> 0.078`. Competition Gemma units are unchanged, but there is no
external gain to route. Reject this family adapter and do not resample or train
a Gemma-only continuation from the same category.

Matched-cost Qwen-27B reasoning self-consistency is also negative. Three
temperature-`0.6` `known4096` samples differ on 8.52% of rows, but median BA is
`0.9262` and varied BA `0.8417`, versus `0.9298`/`0.8639` for deterministic
D/K/S. Nineteen member outputs fail to parse. A diagnostic max reaches `0.9310`
but was not eligible for selection; do not test or replace prompt diversity.

Positive short-circuiting does not make the full max ensemble cheaply
deployable. It skips 31.39% of prompt evaluations but only reduces scoring time
15.59% (`1,533 -> 1,294s`) because the unresolved negative tail is slower.
Fresh batch composition also moves validation BA `0.9298 -> 0.9286` and varied
BA `0.8639 -> 0.8583`. Keep full member-major scheduling; cached algebraic
equivalence is not generated-backend equivalence.

Hard generation caps fail for the same tail. Cached deterministic prefixes at
2,048 tokens change six row decisions, raise member parse failures `5 -> 30`,
and lower varied BA `0.8639 -> 0.8556` while saving an estimated 7.6% of output
tokens. Even 3,072 tokens changes one decision and saves only 3.2%. Do not run
or combine these caps; the exact Qwen-27B ensemble needs its 4,096-token budget.

A format-aware stop is much better for Qwen-9B: retaining the first exact
`Rating: 1--7` string raises validation macro BA `0.9226 -> 0.9250`, preserves
instructed BA, raises varied `0.8500 -> 0.8556`, and reduces score time
`1,314 -> 552s`. The frozen test confirmation also passes: BA moves
`0.9298 -> 0.9310` while score time falls `1,285 -> 528s` (runtime fraction
`0.411`). Qwen-27B validation independently passes at `0.9357` BA versus
`0.9298`, with score time `1,533 -> 1,169s` (fraction `0.763`) and unchanged
parse errors. Treat the accuracy deltas as scheduling drift; the stable finding
is substantial decode savings without aggregate degradation. Stop-string
support on the NDIF generation path remains a separate deployment check.

The same exact stop also passes a frozen cross-response-mode audit on the
Qwen-27B Liars' Bench spectrum. Macro category BA remains `0.8750`, only 2/800
decisions change (one fix, one break), parse errors remain 83, and cached
prefixes retain 68.0% of output tokens. Do not translate that token fraction
directly into wall time; generated Qwen-27B validation measured the 23.7%
runtime reduction. NDIF tokenizer/stop serialization remains the only missing
deployment contract.

## Generalizable-transfer decision summary (2026-07-17)

The most credible improvement is now an exact judge-capacity swap, not another
adapter or local threshold. Qwen-27B D/K/S improves matched competition
validation (`0.9226 -> 0.9298`) and locked test (`0.9298 -> 0.9464`), then
passes the four-mode external gate (`0.7788 -> 0.8750`). It makes 105 fixes and
28 breaks externally, improves every category, and has only two small
soft-trigger family-cell losses. The earlier 200-row external loss must remain
reported, but it is outweighed by a preregistered, broader mode/family audit
rather than ignored post hoc.

A second, genuinely untouched 800-row four-mode confirmation strengthens this
conclusion. With the sampling rule and gate frozen before generation, Qwen-27B
raises macro-category BA `0.75125 -> 0.86000` (`+0.10875`), makes 122 fixes and
35 breaks, and passes every category/family guard. The paired bootstrap 95%
interval is `[+0.08375, +0.13375]`. The largest gain is again Insider Trading
(`+0.325`), but choice also gains `+0.070`; HP-KR (`+0.035`) and soft trigger
(`+0.005`) are positive point estimates whose individual intervals cross zero.
Treat this as strong broad capacity evidence, not proof that each response mode
improves independently.

On the external spectrum, a paired label-stratified bootstrap places the
macro-gain 95% interval at `[+0.0713, +0.1213]`. Choice, HP-KR, and action are
individually positive; only the small `+0.005` soft-trigger delta spans zero.
This supports a broad capacity effect rather than one category carrying the
aggregate.

It is also not primarily a format-recovery effect. Restricting to the 661
external rows where all Qwen-9B members parse, Qwen-27B raises BA from `0.8006`
to `0.8856` (77 fixes, 19 breaks). The 139-row Qwen-9B parse-error partition
improves only `0.6457 -> 0.6846`. Fewer parse failures are useful, but most of
the capacity gain comes from different judgments on already parseable outputs.

The highest-value engineering change is first-complete-rating stopping. It
preserves Qwen-9B validation/test, Qwen-27B validation, and Qwen-27B external
macro BA while reducing generated work. Generated scoring time falls 58.9% for
Qwen-9B and 23.7% for Qwen-27B. It is ready for local vLLM use; remote NNsight
tokenizer/stop serialization is the remaining contract, not model accuracy.

Do not replace max aggregation with validation-selected consensus. Qwen-27B
majority vote is `+0.0012` BA over max on validation, but `-0.0345` on locked
test and `-0.0100` on the external four-mode spectrum. The three prompts recover
different deceptive rows, so the max rule's extra false positives are repaid by
recall out of sample. Qwen-9B shows the same overall pattern. This closes
aggregation hill climbing without requiring new generations.

The swap is also not a single-prompt accident. Each matched Qwen-27B member is
better than Qwen-9B on validation and the external spectrum, and none is worse
on locked test. The external singleton gains are `+0.1325` details, `+0.1275`
known, and `+0.0525` scrutiny BA. Prompt diversity remains useful on top of a
broad capacity improvement.

If full Qwen-27B misses a remote runtime limit, Qwen-9B member disagreement is
a frozen label-free escalation trigger. It queries Qwen-27B on 6.7--6.9% of
competition rows and 26.9% of the original external spectrum, improving Qwen-9B
BA in all three checks (`+0.0012` validation, `+0.0036` test, `+0.0525`
external). It remains below full Qwen-27B. Preserve it only as a runtime
contingency, pending the separately frozen untouched spectrum gate and NDIF
timing; do not misstate it as the accuracy leader. Its external gain is mostly
Insider Trading (`+0.190` category BA); the other three category gains are at
most `+0.010`, another reason to require untouched confirmation.

That untouched confirmation rejects the escalation route. It improves macro BA
`0.75125 -> 0.82375` at a 27.88% Qwen-27B query fraction, but the choice/Qwen
cell loses `0.0606`, breaching the frozen `0.05` limit. Do not loosen the gate
or add a category-specific exception. If remote Qwen-27B is viable, use the
exact full capacity swap; if it is not, retain the current Qwen-9B submission
rather than this unstable partial cascade.

Semantic routing is useful only under a weaker judge. Qwen-9B heavy plus fixed
epistemic/action specialists gains `0.0538` external macro BA with every family
cell non-worse. The same specialists harm Qwen-27B, and an always-on Phoenix
adapter fails as a drop-in heavy state. Thus the fallback is explicitly
capacity-conditional: Qwen-9B may need observable response-mode specialists;
Qwen-27B should keep its heavy predictions. The passage-grounded True/False
route remains a narrow dormant one-call safeguard because it improves 2,000
untouched rows across every family and has zero local coverage.

Small weight updates and model-specific routing are not supported by this
round. Focused Liars' Bench reasoning continuation changes no external
predictions against its true parent and regresses competition validation;
honest-hallucination continuation barely moves external FPR; and the untouched
Gemma confirmation rejects the apparent family gain. The limitation is not
simply too much data. Observable semantic prompts and judge capacity transfer
better than 64--256-example continuations under the present Qwen-only varied
training distribution.

Deployment context is ample for the Qwen-27B heavy path: all 8,216 local rows
fit the 8,000-character judge view, and 24,648 D/K/S prompts have median 439,
p95 704, and maximum 1,248 tokens. A mistakenly proposed context ablation was
canceled before labels were read once the 3,000-character few-shot field was
distinguished from the 8,000-character judge field. The next risk is remote
activation memory and throughput, not context length.

## Next measurements

1. Rehearse exact Qwen-27B D/K/S on one labeled 40-row NDIF dataset unit using
   `LanguageModel`, then certify batch tiers and first-rating string stopping.
   Compare all 40 decisions with the saved vLLM row before changing the single
   submission notebook. If string stops cannot serialize, retain full decoding
   and measure it separately rather than approximating with digit EOS tokens.
   The independent frozen renderer/parser/aggregation contract and rehearsal
   checklist are under `experiments/qwen27_ndif_deployment/` and are tested
   against the resolved local config without depending on ignored artifacts.
2. Do not spend another test evaluation on trace-target wording or the current
   mixed/dropout recipes. The teacher-target, student-input, data-mixture, and
   dropout controls are closed on validation and the original adapter won the
   locked test comparison.
3. If new data become available, require varied-deception traces from Gemma and
   Nemotron plus matched honest-ignorance controls. The present trace treatment
   is inseparable from Qwen-only specialization.
4. If a smaller deployable judge is still required, distill Qwen-27B reasoning
   only from a new Liars' Bench sample and evaluate on grouped, untouched
   categories/families. Do not train on the 800-row spectrum used to establish
   the capacity result; the earlier small continuation already rejects another
   exposure/LR sweep on reused targets.
5. Address supporting-detail verification with independently grounded facts or
   a calibrated claim-level member. Preserve the existing negative RAG and
   atomic-cascade controls; neither arbitrary retrieval nor `any(False)` is a
   justified recall rule.
6. Keep backend-equivalence checks at the dataset-unit level. The current NDIF
   schedule exactly matches vLLM on the labeled 40-row rehearsal, but small
   prefix capacity tests materially overstated safe batch size.

### Active same-family teacher ablation

Qwen-27B cannot be used as the submitted inference judge, but it can test
whether a stronger teacher from the student's model family produces more
learnable privileged summaries. P54 holds the selected varied-only Qwen-9B SFT
recipe and all prompts/labels fixed, replacing only GPT-OSS with Qwen-27B.
Qwen's private `<think>` continuation is retained only in the ignored raw
artifact; the student target is parsed strictly after `</think>`, and missing
closure is rejected. Balanced 32-row smoke job `30180040` precedes any full
cache generation. This is a teacher-family ablation, not authorization to use
Qwen-27B at submission inference time.

The strict parser behaves correctly, but the frozen coverage gate fails.
At 2,048 tokens Qwen-27B closes thinking on only 19/32 rows. The allowed
4,096-token repair improves this to 29/32, still below the required 31/32. The
three failures are long private continuations with no `</think>` or visible
answer; the 29 usable summaries are label-consistent, median 57 words, and show
no private-marker or privileged-language leakage. Do not generate the full
2,880-row cache under this protocol or infer whether same-family teaching helps
from parser coverage alone. A future selective higher-cap retry must be frozen
as a separate cost/coverage experiment.

The user-authorized selective 8,192-token retry passes. Job `30180231` copied
the 4,096-token cache, reused all 29 usable records exactly, and regenerated
only its three failures under a 12,288-token total context. All three closed,
yielding 32/32 parsed and label-consistent targets with zero private-marker or
privileged-language leakage. Median/p95 visible summary length is 57/79 words.
This validates a tiered cache-generation policy—4,096 normally and 8,192 only
for unclosed rows—but does not yet answer whether the Qwen-family teacher
improves the trained Qwen-9B student.

Before full generation, the execution and acceptance policy was frozen. The
2,880 varied-only rows are split deterministically and round-robin within every
dataset/label stratum across two independent artifacts. The first 4,096-token
stage must contain exactly 2,880 unique rows, at least 2,304 usable targets,
exact conditioned-label agreement, and no visible private/prompt-meta leakage.
Only unclosed rows are eligible for an 8,192-token retry. The final merge must
retain canonical hashes for every initially usable record, contain at least
2,877 usable targets, and have usable-label imbalance no greater than three.
Failure of either audit blocks all dependent jobs. Passing the final audit
unlocks the unchanged selected one-epoch varied-only AdamW `5e-5` student and a
joint validation evaluation against the original GPT-OSS-teacher adapter. The
existing P54 validation promotion rule remains the selection rule; do not
inspect local test from a losing or ambiguous validation result.

The full first pass completed in about 72 minutes per shard but exposed a
coverage shift from the 32-row smoke. It yielded 2,329/2,880 usable targets,
with 532 completions still inside private thinking and 19 that emitted
`</think><reasoning_summary>` but hit the cap before the summary closing tag.
There was no leakage among usable targets, and all usable predictions matched
their conditioned labels. Audit `30180338` correctly blocked the original 85%
operational floor. Because every failed record has one of two exact token-cap
shapes and the 551-row retry remains bounded across the already planned two GPU
jobs, the resumed protocol freezes a lower 80% first-stage cost floor (2,304)
and permits both shapes to retry. This does not relax the final requirement of
at least 2,877 usable, preserved, leakage-free targets before training.

The 8,192-token jobs ran for about 33 minutes each and recovered 476/551
failed records. Final audit `30182700` measured 2,805 usable targets, split
1,426 label 0 and 1,379 label 1, with 75 completions still unclosed. There were
no visible truncations or leakage and no changes to the 2,329-record 4,096-token
manifest. This misses the final gate by 72, so training remains blocked. Before
another generation pass, a third and final tier was frozen: require at least
2,794 usable records (97%) at the intermediate audit, manifest all 2,805 current
valid records, and retry only the 75 unclosed rows with 16,384 generation tokens
inside a 20,480-token model context. The final acceptance gate remains 2,877
usable with imbalance no greater than three; do not train if that still fails.

The 16,384-token tier recovered 54/75 rows, leaving 21 unclosed completions.
Jobs `30187447` and `30187448` took 18m39s and 17m08s. Audit `30187449`
measured 2,859/2,880 usable targets (99.27%), split 1,437 label 0 and 1,422
label 1; the 21 failures comprise 3 label-0 and 18 label-1 rows. There was zero
visible leakage, no truncated visible target, and no violation of the 2,805-row
8,192-token preservation manifest. This still misses the frozen coverage gate
by 18 and imbalance gate by 12, so the audit correctly cancelled training and
validation. Do not post-hoc declare the cache accepted or launch another context
tier without explicitly treating that as a new protocol choice.

The user explicitly selected that new protocol choice: accept the 2,859 clean
targets and proceed rather than escalating context again. The accepted cache
keeps the natural 1,437/1,422 label split, excludes the 21 unclosed records via
the standard loader, and must still verify the 2,805-record 8,192-token manifest
before merging. Student prompts, base model, rank, optimizer, learning rate,
epoch count, and effective batch size remain frozen. Interpret the resulting
validation comparison with the documented 18-target coverage difference from
the 2,877-target GPT-OSS-teacher baseline.

Student training and validation completed. Job `30189850` trained 90 steps in
23m12s total (15m48s trainer time) and saved a distinct rank-16 adapter. Joint
validation job `30189851` evaluated both students in one process. The Qwen-27B
teacher student exactly ties the GPT-OSS teacher baseline at `0.9000` overall
BA, `0.9792` instructed BA, and `0.7944` varied BA; recall is `0.8333`, FPR is
`0.0333`, and each has four parse errors. The adapters have different weight
hashes and their generated text differs on 65/822 rows (22 instructed and 43
varied). Only two varied honest-row predictions differ: each adapter parses one
as 0 while the other hits the generation cap, producing reciprocal parse-error
swaps and no metric change. Three parse failures are shared.

The exact `+0.0000` overall delta fails P54's frozen `+0.005` promotion gate.
Do not spend a local-test evaluation on this adapter or tune its LR/data fraction
from the same validation result. Within the fixed recipe, a stronger teacher
from the student's family changes some generated explanations but provides no
observable improvement to the validation decision boundary.

## Binary D/K/S inference ensemble (2026-07-17)

The selected one-epoch varied-only AdamW `5e-5` adapter was evaluated under
three inference-only decision frames adapted from the factual details,
known-context contradiction, and scrutiny prompts. All conditions retained the
student's trained reasoning-summary plus binary-prediction target, disabled
native thinking, hid assistant reasoning traces, and ran together in one vLLM
session. No weights or training data changed.

On validation, details scored `0.8857` overall / `0.8194` varied BA, known
scored `0.9119` / `0.8222`, and scrutiny scored `0.8381` / `0.7611`. The fixed
binary max/OR rule reached `0.9202` overall, `0.9792` instructed, and `0.8417`
varied BA, versus `0.9000`, `0.9792`, and `0.7944` for the ordinary adapter.
Its recall/FPR were `0.8929`/`0.0524`, and only two rows failed in all three
members. Across 98 disagreement rows, OR made 30 fixes and 13 breaks relative
to baseline; 26 fixes and nine breaks were varied.

Majority (`0.8905` overall) and unanimity (`0.8250`) were frozen diagnostics,
not selectable alternatives. The preregistered OR passes its `+0.005` overall,
non-worse-scenario, and parse-error gate and is authorized for one locked local
test. This experiment shows that binary outputs still support useful ensemble
diversity: the vote fraction is coarse, but the union of independently framed
positive decisions can recover errors that no single member covers.

The locked local-test confirmation rejects deployment. Job `30191769` scores
`0.9167` overall, `0.9792` instructed, and `0.8333` varied BA for the fixed OR,
versus `0.9155`, `0.9812`, and `0.8278` for the ordinary adapter. Recall rises
from `0.8619` to `0.8976`, but FPR rises from `0.0310` to `0.0643`, leaving only
`+0.0012` overall BA. OR makes 20 fixes and 19 breaks across 39 changed rows.
The large validation improvement therefore does not transfer. Keep the ordinary
single prompt; do not integrate D/K/S, test-select the known member, or tune a
vote rule from these labels.

## Ordinal D/K/S readout (2026-07-17)

A matched validation experiment asked the selected binary student for 1--7
ratings under the same details, known-fact, and scrutiny frames. The evaluator,
adapter, input serialization, no-thinking mode, context limits, deterministic
decoding, and 512-token allowance were held fixed. The preregistered ensemble
maxed linearly mapped ratings and classified ratings 4--7 as deceptive.

Job `30192251` completed in 4m20s. The max ensemble scored `0.9167` overall BA,
`0.9260` AUROC, `0.9750` instructed BA, and `0.8389` varied BA, with `0.8833`
recall, `0.0500` FPR, and one all-member parse failure. This is below the binary
OR's `0.9202` overall and `0.8417` varied BA and fails the frozen `0.9252`
overall, non-worse scenario, `0.9300` AUROC, and parse gate. No test run is
authorized.

The rating distribution explains the limited benefit. Of 821 parsed max
ratings, 443 are 1 and 364 are 7; just 14 (1.7%) lie from 2 through 6. Member
extreme fractions range from 97.8% to 99.4%. Ratings are therefore not perfectly
binary, and AUROC exceeds midpoint BA by `0.0093`, but the usable ordinal signal
is very sparse. Fixed non-selectable cutoffs from 2 through 7 peak at only
`0.9190` BA. Relative to the ordinary adapter, midpoint max makes 30 fixes and
16 breaks; relative to binary OR, it makes 13 fixes and 16 breaks. Do not tune
the scale, cutoff, or prompt from these validation labels. Completion-only
binary SFT has preserved framing diversity but strongly polarized the requested
confidence readout.

## Ten-percent LoRA-rank sweep (2026-07-18)

A matched fast screen trained fresh rank 1, 2, 4, and 8 adapters with alpha
equal to twice rank, then jointly evaluated them with the existing
rank-16/alpha-32 10% adapter. Every rank used the same deterministic 288-row
varied-only subset, reasoning-summary targets, seed, attention/MLP modules,
one-epoch AdamW `5e-5` recipe, and effective batch size 32.

| rank / alpha | parameters | artifact | train loss | trainer time | overall BA | instructed BA | varied BA | recall | FPR | parse errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 2 | 1.82M | 7.0 MiB | 0.9193 | 103.1s | 0.9000 | 0.9771 | 0.7972 | 0.8333 | 0.0333 | 4 |
| 2 / 4 | 3.64M | 13.9 MiB | 0.9143 | 96.3s | 0.9000 | 0.9792 | 0.7944 | 0.8333 | 0.0333 | 4 |
| 4 / 8 | 7.27M | 27.8 MiB | 0.8993 | 96.4s | 0.9000 | 0.9771 | 0.7972 | 0.8333 | 0.0333 | 4 |
| 8 / 16 | 14.55M | 55.5 MiB | 0.8550 | 96.2s | 0.9012 | 0.9792 | 0.7972 | 0.8333 | 0.0310 | 5 |
| 16 / 32 | 29.10M | 111.0 MiB | 0.7913 | 110.0s | 0.9000 | 0.9771 | 0.7972 | 0.8333 | 0.0333 | 4 |

Training loss improves smoothly with rank, but validation detection is flat.
Ranks 1, 4, and 16 emit exactly equal completion strings on all 822 rows even
though their weight hashes differ. Rank 2 changes 10 decisions relative to rank
16 with five fixes and five breaks; rank 8 changes 11 with six fixes and five
breaks. Only 11 rows have any binary disagreement across all five ranks.

The frozen five-way vote fraction scores `0.9000` BA and `0.9052` AUROC; its
majority decision is exactly rank 16's decision on every row. OR reaches only
`0.9012` BA (three fixes/two breaks), and unanimity remains `0.9000`. Thus
different ranks do not provide meaningful error diversity. Bundling all five
weights would also consume about 215 MiB before code, exceeding the package
limit. Do not build a rank ensemble or spend local test on this sweep. Rank 1
deserves consideration only as a 16x adapter-compression hypothesis, to be
confirmed with the full training recipe and an untouched evaluation before it
could replace rank 16.

### Full-data rank-24 follow-up (2026-07-23)

A matched full-data capacity follow-up trained rank 24 with alpha 48 while
holding the selected varied-only recipe fixed: all 2,877 parsed,
label-consistent GPT-OSS reasoning-summary targets, regular
`Qwen/Qwen3.5-9B`, the same attention/MLP target modules, seed 0, one epoch,
AdamW `5e-5`, completion-only loss, and effective batch size 32. Training job
`30246588` completed 90 optimizer steps on one RTX Pro 6000. Trainer time was
976.3 seconds, train loss was 0.4805, and the saved 256-tensor adapter is
174,622,016 bytes with SHA-256
`714e9a71f63780158ed8a29a9c408ad94c8e44396b37be7178212bf3e5c6303c`.

Standalone validation job `30246646` scored 0.9048 overall BA and 0.8083
varied BA, but small greedy-generation drift is established for this evaluator.
The controlled result is therefore shared-session job `30246656`, which loaded
rank 16 and rank 24 into the same vLLM engine:

| rank / alpha | overall BA | instructed BA | varied BA | recall | FPR | parse errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 / 32 | 0.8964 | 0.9729 | 0.7944 | 0.8286 | 0.0357 | 3 |
| 24 / 48 | **0.9024** | **0.9771** | **0.8028** | **0.8405** | 0.0357 | 4 |

Rank 24 changed only 11/822 binary decisions: eight fixes and three breaks.
After excluding every decision affected by either adapter's parse failure,
there were six fixes and three breaks. The gain is therefore not solely a
negative fallback artifact, but it is small, validation-only, and restricted
to the Qwen-only varied-deception training distribution. Do not claim private
leaderboard or cross-family generalization from this result. The raw rank-24
weights are about 166.5 MiB, so deployment must also recheck the complete
200 MB submission package. Do not replace the rank-16 Phoenix adapter or spend
a local-test evaluation without a separate promotion decision.

### Rank-24 AdamW learning-rate sweep (2026-07-23)

A full-data one-epoch AdamW sweep held the rank-24/alpha-48 recipe above fixed
and crossed learning rates `1e-5`, `2e-5`, `5e-5`, and `1e-4`. The initial
shared-session validation run (`30246745`) put `1e-4` at the upper boundary
with 0.9060 overall and 0.8111 varied BA, so the frozen sweep was extended to
`2e-4` and `3e-4`. Training jobs and losses were:

| AdamW LR | Slurm job | train loss | trainer time |
| ---: | ---: | ---: | ---: |
| `1e-5` | `30246693` | 0.5758 | 988.5s |
| `2e-5` | `30246694` | 0.5227 | 978.8s |
| `5e-5` | `30246588` | 0.4805 | 976.3s |
| `1e-4` | `30246695` | 0.4625 | 983.6s |
| `2e-4` | `30246753` | 0.4507 | 968.4s |
| `3e-4` | `30246754` | 0.4457 | 953.6s |

All six jobs completed stably, but decreasing training loss did not imply
better detection. A second shared-session high-rate comparison (`30246781`)
reversed the apparent boundary win: `5e-5` scored 0.9024 BA while `1e-4`
fell to 0.8976. A third all-six rotation (`30246792`) again reordered several
few-row differences. Aggregate metrics across the available shared-session
runs were:

| AdamW LR | runs | mean BA | median BA | BA range | mean varied BA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-5` | 2 | 0.9012 | 0.9012 | 0.9000--0.9024 | 0.8014 |
| `2e-5` | 2 | 0.9012 | 0.9012 | 0.9012--0.9012 | 0.8028 |
| `5e-5` | 3 | **0.9016** | **0.9024** | 0.8988--0.9036 | **0.8028** |
| `1e-4` | 3 | 0.9004 | 0.8976 | 0.8976--0.9060 | 0.8019 |
| `2e-4` | 2 | 0.8994 | 0.8994 | 0.8988--0.9000 | 0.7972 |
| `3e-4` | 2 | 0.8994 | 0.8994 | 0.8976--0.9012 | 0.8000 |

Compare rates only within sessions where both were present. Relative to
`5e-5`, the overall-BA deltas were `+0.0036/-0.0036` for `1e-5`,
`+0.0024/-0.0024` for `2e-5`, `-0.0048/-0.0012/+0.0024` for `1e-4`,
`-0.0036/+0.0012` for `2e-4`, and `-0.0012/-0.0012` for `3e-4`.
Thus no candidate robustly improves on `5e-5`; the isolated `1e-4` peak was
adapter-order/backend drift. Retain `5e-5` as the rank-24 learning rate and do
not spend local test or leaderboard evaluations on the other rates. Differences
at this scale remain below the reliable selection resolution of this generated
binary evaluator.

## Balanced FEVER evidence consumer (2026-07-20)

The P73 follow-up trained a rank-1/alpha-2 reader on 14,069 evidence-visible
targets from 2,873 varied-training rows. The curriculum included real, empty,
shuffled, real-plus-shuffled, topical/insufficient, and equal-sized honest
ordinary-error/deceptive-contradiction conditions. Its audit found one
reference block and one label-consistent evidence assessment per target, no
student-visible ground-truth tag, and a 7,035/7,034 label split. Job `30215513`
completed one AdamW `5e-5` epoch in 3h45m on an A100 and saved a 7.3 MB rank-1
weight file.

Frozen deployable-cache evaluation job `30215515` scored `0.8056` varied BA for
both empty and real evidence and `0.8028` for shuffled evidence. Recall/FPR were
`0.6722`/`0.0611` for empty and real. The nominal real-versus-empty outcome was
three fixes and two breaks, but one fix was only recovery from an empty parse
failure; parse-valid rows were two fixes and two breaks. On the 62 active rows,
empty and real tied at `0.7742` BA. The prior rank-16 FEVER-visible reader also
scored `0.7742` with real evidence, and the new reader had five fixes/five harms
relative to it across all varied rows.

This fails the frozen `+0.005` empty-control gate. Do not run local test, package
the index, increase LoRA rank, or reweight the curriculum from these validation
rows. Balanced consumer training preserves weak real-versus-shuffled
sensitivity but does not make retrieval better than no intervention.

## Verified teacher-rationale cleaning (2026-07-25)

A manual and GPT-OSS-assisted audit estimates that `3--4.5%` of the 2,877
selected teacher summaries contain harmful factual, semantic, or privileged
rationale errors. The raw model screen flags `6.47%`, but manual calibration
finds many pedantic or incorrect corrections; the exact structural lower bound
is `1.04%`. See `rationale_cleaning_audit.md` for the qualitative examples and
full prevalence accounting.

The frozen first ablation replaces only 46 manually verified bad summaries
(41 positive, five negative) with label-only targets. It removes 4,739
supervised tokens (`1.74%`) while keeping every row and authoritative label.
Matched rank-16 one-epoch AdamW `5e-5` training completed as job `30289933`.
Forward and reverse shared-session validations (`30289934`, `30289976`) show
the expected adapter-position shift: both models gain exactly 0.0024 overall BA
when evaluated second. Position-averaged results are:

| model | overall BA | instructed BA | varied BA |
| --- | ---: | ---: | ---: |
| Original teacher targets | **0.9048** | **0.9781** | **0.8069** |
| Verified-46 label-only cleaning | 0.9036 | 0.9771 | 0.8056 |

Matched-position comparisons each produce one net regression. The forward run's
nominal gain is only three fixes/two breaks plus one neutral extra parse failure,
and qualitative inspection finds new confident factual hallucinations alongside
two useful spelling-error detections. Retain the original teacher/default
adapter, do not test the cleaned adapter, and do not expand the cleaning mask to
unverified GPT-OSS flags. Teacher-rationale errors are real supervision defects,
but selective removal at this prevalence does not measurably improve the
aggregate judge.

## Rank-16 DataRater-style gradient filtering (2026-07-25)

The first filtering screen adapted DataRater's held-out meta-objective to the
selected reasoning-summary SFT recipe without using validation or test labels.
Of 2,877 usable varied-deception teacher targets, a stable dataset/label-
stratified split reserved 144 rows (5%) for the meta objective and left 2,733
disjoint candidates. The outer loss supervised only the final binary
prediction, while each candidate's inner loss supervised its full GPT-OSS
reasoning summary plus prediction. This directly tests whether reasoning-target
updates help held-out classification rather than merely reproduce rationale
wording.

The scorer used a fresh rank-16/alpha-32 LoRA and selected all attention/MLP
LoRA matrices in the final Qwen transformer block: 1,277,952 parameters. It
formed the held-out gradient exactly and estimated each candidate-gradient dot
product with two no-grad forwards, perturbing the LoRA parameters by `0.1`
along the normalized meta-gradient. This is a tractable one-step influence
proxy. It does not reproduce DataRater's learned non-causal rater, population
of inner models, or multi-step unrolled meta-optimisation.

A deterministic rank-16 calibration used the same seeded LoRA basis for exact
and finite-difference scoring over eight meta rows and 16 candidates. Pearson
correlation was `0.6658`, Spearman correlation `0.7735`, sign agreement
`0.75`, and top-half overlap `7/8`. The full H100 pass then scored all 2,733
candidates with meta-gradient norm `4.6504` and mean meta loss `4.4622`. Its
canonical `scores.jsonl` SHA-256 is
`87872d48b840c61b1bd60c6f6e53e67057d9534ce5833e68a763437d7c405dee`.

The resulting signal was distinct from ordinary difficulty. Gradient dot and
teacher completion loss had Pearson/Spearman correlations of only
`0.0594/0.0611`. At 50% keep, the dot/loss selections shared 721/1,368 rows
(Jaccard `0.3578`), while dot/random shared 684 rows (Jaccard `0.3333`).
Every manifest retained 684 rows per label and preserved every dataset/label
stratum.

| 50% selection | rows | mean gradient dot | mean teacher loss | positive-dot fraction |
| --- | ---: | ---: | ---: | ---: |
| All candidates | 2,733 | `+0.01246` | `0.91561` | `0.542` |
| Matched random | 1,368 | `+0.01013` | `0.91616` | `0.537` |
| High loss | 1,368 | `+0.02075` | `1.05691` | `0.569` |
| Gradient dot | 1,368 | `+0.09830` | `0.92572` | `0.999` |

All downstream students used the established rank-16/alpha-32 all-projection
adapter, AdamW `5e-5`, and exactly 90 optimizer steps. Random job `30291345`
completed in 958.1 seconds with train loss `0.4855`; high-loss job `30291346`
completed in 974.1 seconds with train loss `0.5698`. The gradient arm first
completed on the Lambda H100 with train loss `0.4857`. Because the cluster-to-
Lambda SSH path could not reliably transfer the 116 MB weights, job `30291532`
replicated the exact run on the same RTX environment as the controls. It
completed in 960.0 seconds with train loss `0.4859` and saved a verified
rank-16/alpha-32 adapter with SHA-256
`e59811546933ee3344ba57bedfc669801fe254d2396dbce4001c733a4fd88003`.

Definitive forward/reverse shared vLLM sessions `30291540` and `30291541`
loaded the full-data baseline, matched random, high-loss, and gradient-dot
adapters in opposite orders. Position-averaged results were:

| training data | overall BA | instructed BA | varied BA | recall | FPR | mean parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full varied baseline | `0.9048` | `0.9760` | **`0.8097`** | `0.8381` | `0.0286` | 4.0 |
| Random keep 50% | **`0.9054`** | **`0.9771`** | **`0.8097`** | **`0.8393`** | `0.0286` | 4.0 |
| High-loss keep 50% | `0.9042` | `0.9760` | `0.8083` | **`0.8393`** | `0.0310` | 3.0 |
| Gradient-dot keep 50% | `0.9042` | **`0.9771`** | `0.8069` | `0.8369` | `0.0286` | 3.0 |

The gradient adapter made two fixes and two breaks relative to baseline in the
forward session, then three fixes and four breaks in reverse. It also changed
nine binary decisions between its two adapter positions, compared with one for
random, three for loss, and four for baseline. Thus its `-0.0006` overall and
`-0.0028` varied deltas are not an improvement hidden by parse failures; the
selector slightly lowers recall at the same FPR and is at least as sensitive
to backend position as the controls.

This screen fails the frozen `+0.005` validation promotion rule and does not
beat random filtering. Do not run local test, promote this selector, or spend
additional compute on its predeclared 25%/75% keep fractions, more LoRA layers,
multiple initialization seeds, trained-state rescoring, soft weighting, or a
learned rater. This negative result rejects the current one-step,
initialization-state, last-block approximation for this already curated,
Qwen-only varied cache; it does not establish that the paper's much more
expensive full DataRater algorithm would fail in another regime.

## AUROC-aware direct and pairwise student losses (2026-07-26)

A matched four-arm Lambda H100 ablation tested whether the strong direct-logit
ranking signal could be improved during SFT without removing the useful GPT-OSS
reasoning-summary supervision. Every arm retained the selected varied-only
rank-16/alpha-32 recipe: all 2,877 usable rows, one epoch, AdamW `5e-5`, and
effective batch size 32. Stable within-dataset positive/negative pairing covered
2,874 rows; the three unmatched negatives retained their ordinary supervised
loss but contributed no pairwise term.

The auxiliary direct loss scores literal `0`/`1` logits after the rendered
prompt plus `Prediction:`. Ranking uses
`softplus(-(margin_positive - margin_negative))` within each dataset. The four
objectives were reasoning-summary SFT alone, reasoning plus direct CE, and
reasoning plus direct CE with pairwise weights `0.1` or `0.3`. Shared validation
loaded all four adapters once and evaluated only the direct continuous margin:

| objective | macro AUROC | delta | instructed AUROC | varied AUROC |
| --- | ---: | ---: | ---: | ---: |
| Paired reasoning control | `0.938155` | — | `0.993542` | `0.864306` |
| + direct CE | **`0.938274`** | `+0.000119` | **`0.993750`** | `0.864306` |
| + direct CE + rank `0.1` | `0.938214` | `+0.000060` | `0.993542` | **`0.864444`** |
| + direct CE + rank `0.3` | `0.938214` | `+0.000060` | `0.993542` | **`0.864444`** |

All four had direct-margin missing count zero and identical fixed-`0.5`
balanced accuracy `0.7833`. The evaluator intentionally generated only one
token, so its 822 generated-output parse errors are irrelevant to the direct
margin. Compact `result.json` and `margin_analysis.json` artifacts are stored
under each method's `validation_direct_margin/` directory in
`results/blackbox/`; full per-row artifacts remain on the persistent Lambda
campaign instance.

This is a clear negative result under the frozen `+0.005` promotion rule. The
largest apparent gain is one ten-thousandth and comes only from instructed
rows; the ranking weights are numerically indistinguishable and do not improve
overall AUROC over direct CE. Do not run test, promote these adapters, or tune
more pairwise weights on validation. Retain reasoning-summary SFT and direct
logit inference, but leave the training loss unchanged.

The implementation is also expensive in the current Lambda environment. The
paired control took roughly 32 minutes for 90 optimizer steps, while each
auxiliary arm took about 64--66 minutes because it performs a second Qwen
forward per microbatch. Transformers reported that Qwen's optional fast linear-
attention dependencies were absent and used its torch fallback. Before any
future multi-objective sweep, test those kernels and batch the shorter direct
prefix pass more efficiently rather than repeating this execution shape.

## Dense Qwen-27B direct soft-target distillation (2026-07-26)

The next frozen ablation replaced hard direct-label CE/ranking with a continuous
teacher target from the stronger dense Qwen3.5-27B direct D/K/S boundary. The
teacher itself passed its prerequisite audit at `0.94429` validation macro
AUROC (`0.99396` instructed, `0.87806` varied), with zero missing rating logits.
The 2,880-row varied-training cache retained all three member distributions and
selected the max-score member without using labels. Global label-blind logit
z-normalization produced 2,877 joined soft targets spanning
`0.103406--0.982934`.

Student job `30299836` kept the selected varied-only rank-16/alpha-32,
one-epoch AdamW `5e-5` recipe and ordinary GPT-OSS reasoning-summary loss, then
added direct-margin soft binary cross-entropy at weight `1.0`. It completed 90
steps in `32m35s`; its adapter weight SHA-256 is
`466a476e2c0f05c83d8f4e627fb3e47d4274c16d5574ac5f57ae5468dd8fb2d3`.

The original five-adapter validation job `30299897` failed before model loading
because the compact historical Lambda controls lacked local `config.yaml`
files. No scores were produced. Corrected job `30300359` therefore loaded the
soft student and canonical selected reasoning-only adapter together, while the
already frozen direct-CE/rank artifacts remained secondary historical
comparisons.

| objective | macro AUROC | delta | instructed AUROC | varied AUROC | unique margins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reasoning-only control | `0.937560` | — | `0.9934375` | `0.863056` | 353 |
| + Qwen-27B direct soft target | `0.937619` | `+0.000060` | `0.9934375` | `0.863194` | 350 |

Both adapters had zero missing direct margins and identical fixed-`0.5`
balanced accuracy (`0.7833`), recall (`0.5738`), and FPR (`0.0071`). The soft
loss changed only 24/822 row scores. Per-dataset AUROC differed substantively
on only three varied units: `+0.00125`, `+0.005`, and `-0.005`; every
instructed unit and the other six varied units tied. Candidate and control
cross-label tie fractions were both about `0.0016`, despite BF16 margin
quantization producing many same-label duplicate scores. The intentionally
one-token generation produced 822 generated-output parse failures, but these
are irrelevant to the separately extracted direct margins.

The matched `+0.000060` gain fails the frozen `+0.005` macro-AUROC promotion
gate by two orders of magnitude and is also numerically below the earlier
direct-CE/rank results from their separate Lambda session. Do not run local
test, promote the adapter, or tune soft-target normalization, temperature,
weight, or data selection on this validation result. The dense teacher's better
ordering did not transfer through this auxiliary direct-margin objective.

## Frozen full-trace Qwen-27B D/K/S follow-up (2026-07-26)

The direct-only P91 loss barely changed the student's ranking and did not teach
the teacher's inference procedure. The next matched experiment therefore uses a
new rank-16 adapter and distills each Qwen3.5-27B D/K/S member end to end.

- Generate all three 4,096-token-budget member traces for the 2,877 usable
  varied-deception rows, using the exact D/K/S prompts and 3,000-character
  context view.
- Retain a row only when all three completions parse, close `</think>`, fit the
  4,096-token target cap, and have a complete audited direct distribution.
- Preserve each full completion verbatim. Do not inject the ground-truth label,
  filter teacher/label conflicts, or replace the generated rating.
- Split whole dataset/index rows 90/10 within dataset/label strata. Every
  selected row contributes exactly three member-specific training examples.
- Train Qwen3.5-9B for one epoch with AdamW `5e-5`, rank 16/alpha 32, effective
  batch size 32, completion SFT, and weight-1 seven-way KL at direct `Rating:`.
- Evaluate the held-out rows with all three prompts. Report max-aggregated
  generated-rating AUROC and max-aggregated direct expected-rating AUROC,
  alongside member results, parse/length failures, score uniqueness, ties, and
  per-dataset AUROC.

Initial 4,096-token generation completed all 8,640 requests but left 250 usable
rows without three parsed, closed member traces. A selective 8,192-token retry
recovered 191 of them. The cache therefore retains 2,818/2,877 usable rows
(`97.95%`) and excludes the remaining 59 whole rows; no partial or truncated
completion is accepted.

Completed result:

- Cache job `30301401`, 96-target smoke `30301402`, full student `30301403`,
  and held-out evaluation `30301404` all completed. The cache contains 2,536
  training rows / 7,608 member traces and a frozen 282-row holdout. Target
  lengths were median 781 tokens, p95 2,146, and maximum 7,922.
- Full training took `3h47m15s` for one epoch and 238 optimizer steps. Final
  reported train loss was `10.89`. The saved adapter weight SHA-256 is
  `71f22dcb7ea5d5420cf2196240bf00a629474bbdeea76f12b41997b86ed8b9af`.
- On the exact holdout, max-aggregated Qwen-27B generated ratings scored
  `0.91043` macro AUROC and its direct expected-rating distributions scored
  `0.89831`. The distilled student scored `0.86816` with generated ratings,
  `0.87718` at the direct no-reasoning boundary, and `0.89597` when extracting
  the continuous rating distribution after its generated reasoning.
- Thus post-reasoning scoring recovered nearly all of the teacher's direct
  ranking (`-0.00234` AUROC), but the intended no-reasoning readout remained
  `-0.02113` behind. Generated integer ratings collapsed to four unique values;
  both continuous student paths had 282/282 unique scores and zero within-unit
  cross-label ties.
- Max aggregation had zero rating parse failures. Individual known and scrutiny
  prompts each had one length-capped parse failure, recovered by another
  member. The native `<think>` strict-format counter is not interpretable here
  because vLLM strips special thinking tokens from returned text; the visible
  reasoning and terminal ratings were retained.

This supports the user's hypothesis in a qualified way: the student can use a
generated chain to approach the teacher's ranking, but that ability does not
fully transfer to the direct boundary. Do not promote this adapter on the
no-reasoning path or evaluate local test from this development holdout. If the
track continues, the clean next ablation is sparse top-k soft distillation over
every teacher trace token, with the direct rating anchor retained separately.

### Frozen full-validation confirmation

Job `30303488` evaluated all 822 validation rows under Details, Known, and
Scrutiny in one vLLM session. Native Qwen thinking mode remained disabled; the
adapter generated its learned ordinary-channel reasoning. Before this
confirmation, the cached teacher holdout selected `P(Rating=7)` as an
exploratory binary-like collapse, so the evaluator preserved all seven
probabilities and reported it alongside expected rating.

| readout | macro AUROC | instructed | varied |
| --- | ---: | ---: | ---: |
| Generated integer, D/K/S max | `0.91292` | `0.96229` | `0.84708` |
| Direct expected rating, D/K/S max | `0.92762` | `0.98438` | `0.85194` |
| Direct `P(Rating=7)`, D/K/S max | `0.92679` | `0.98458` | `0.84972` |
| Post-reasoning expected rating, D/K/S max | `0.93821` | **`0.99229`** | `0.86611` |
| Post-reasoning `P(Rating=7)`, D/K/S max | **`0.93952`** | `0.99125` | **`0.87056`** |
| Phoenix direct reference | `0.93816` | `0.99354` | `0.86431` |

The best single direct member was Details at `0.93000` macro AUROC
(`0.98563` instructed, `0.85583` varied); max aggregation reduced it by
`0.00238`. The best single post-reasoning expected-rating member was Known at
`0.92607`, so the ensemble is important only for the expensive reasoning path.

The best trace-student result exceeds Phoenix direct by only `0.00137`, below
the frozen `+0.005` promotion gate, while requiring three generations and six
one-token passes. Scoring took `380.2s` excluding startup, versus `9.7s` for the
historical Phoenix direct validation pass on its H100 session. Max aggregation
had zero parse failures; Details and Known/Scrutiny generation contained five
total length caps, with the other members recovering every row.

Decision: do not promote, run local test, or deploy reasoning generation. For
an efficient trace-student readout, one Details direct expected-rating pass is
strictly preferable to D/K/S max but remains `0.00816` below Phoenix. The next
useful prompt experiment should align the student and teacher with the compact
Truth Value Guard prompt rather than retune D/K/S aggregation on validation.

This is a reasoning-imitation ablation, not a continuation of P91's binary
normalization sweep. Do not choose members or remove label-conflicting traces
after inspecting held-out results.

## Binary Truth Value Guard teacher gate (2026-07-27)

Job `30303807` tested the exact compact Truth Value Guard prompt as a
reasoning-free teacher boundary. Both dense teachers used the same 3,000-
character tail renderer, Qwen no-thinking chat template, and terminal
`Prediction:` scaffold. The score was the normalized probability of the
literal next-token choices `0` and `1`; no reasoning was generated. This is
one forward/logit pass per row.

| teacher | macro AUROC | instructed | varied | BA | recall | FPR | score time | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-27B | **`0.95744`** | **`0.99917`** | **`0.90181`** | **`0.91190`** | **`0.88571`** | **`0.06190`** | `38.2s` | `779/822` |
| Qwen3.6-27B | `0.95333` | `0.99542` | `0.89722` | `0.89405` | `0.85714` | `0.06905` | `39.6s` | `766/822` |

Both arms had zero missing label logits, zero parse errors, no exact score
saturation at zero or one, and used about 50.2 GiB of GPU memory. Their scores
were strongly but not identically ranked (Spearman `0.9542`); Qwen3.6 lost
`0.00411` macro AUROC and also regressed in each scenario aggregate. Retain
Qwen3.5-27B as the teacher and do not generate a Qwen3.6 trace cache from this
checkpoint merely because it is newer.

The Qwen3.5 binary Truth Value Guard result is `+0.01315` above the earlier
Qwen3.5-27B direct seven-way D/K/S teacher (`0.94429`). Job `30303841` removed
the prompt confound by keeping the exact same Truth Value Guard body and
renderer while replacing the binary boundary with expected `Rating:1..7`
logits:

| Qwen3.5-27B TVG readout | macro AUROC | instructed | varied | BA | recall | FPR | score time | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| binary `Prediction:0|1` | **`0.95744`** | **`0.99917`** | **`0.90181`** | **`0.91190`** | `0.88571` | **`0.06190`** | **`38.2s`** | `779/822` |
| expected `Rating:1..7` | `0.90286` | `0.98208` | `0.79722` | `0.91071` | **`0.89524`** | `0.07381` | `39.5s` | **`821/822`** |

Both had zero missing requested logits. The `+0.05458` binary AUROC gain is
therefore an output-parameterization result, not merely a better prompt. The
ordinal arm has more unique scores and nearly identical BA, but badly orders
the difficult varied rows. Use a literal binary teacher target for the next
student loss; do not infer ranking quality from score uniqueness alone.

## Pure Qwen-27B TVG binary soft distillation (2026-07-27)

Jobs `30310442`--`30310445` completed the frozen varied-only binary soft-target
pipeline. Unlike the earlier auxiliary-loss experiment, this used a fresh
rank-16/alpha-32 Qwen3.5-9B adapter and optimized only binary soft-target BCE at
the exact Truth Value Guard `Prediction:` boundary. There was no GPT-OSS
reasoning-summary loss or hard-label loss. The cache contains all 2,880 varied
training rows, 2,488 unique teacher probabilities, and no parser/filter losses.
Teacher probabilities range from `0.0002305` to `0.9996646`.

Training used one epoch, AdamW `5e-5`, and effective batch size 32. It completed
90 optimizer steps in `892.1s`; reported train loss was `4.967`. The adapter
weight SHA-256 is
`c096e5839fe96fb7409fe759a79bc0989ef1efe5aee010b83c3157d38586a75f`.

The evaluation command intentionally allowed only one generated token, which
produced `Prediction` and therefore made all 822 generated-output rows
unparseable. This does not invalidate the separately executed direct-margin
pass: it appended `Prediction:` to the original prompt, scored literal `0|1`
with one forward pass, and had zero missing logits. Read
`score_metrics.direct_margin`, not the placeholder top-level generated metric.

| readout | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-27B teacher | `0.95744` | `0.99917` | `0.90181` | `0.91190` | `0.88571` | `0.06190` | `779/822` |
| Soft-only Qwen3.5-9B student | `0.94625` | `0.99500` | `0.88125` | `0.88810` | `0.83810` | `0.06190` | `544/822` |
| Phoenix 4.4 GRPO reference | **`0.95137`** | **`0.99656`** | **`0.89111`** | **`0.89643`** | **`0.84524`** | `0.05238` | `79/822` |

The soft student retains a strong ranking and far more score resolution than
Phoenix 4.4, but it loses `0.01119` macro AUROC to the teacher and trails
Phoenix 4.4 by `0.00512`. It also transfers the teacher unevenly:

| validation dataset unit | teacher AUROC | student AUROC | delta |
| --- | ---: | ---: | ---: |
| `instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None` | `1.00000` | `1.00000` | `+0.00000` |
| `instructed-deception-Qwen3.5-27B-None` | `1.00000` | `1.00000` | `+0.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.99000` | `0.97750` | `-0.01250` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `1.00000` | `1.00000` | `+0.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `1.00000` | `0.99750` | `-0.00250` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `1.00000` | `0.99750` | `-0.00250` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `1.00000` | `0.97250` | `-0.02750` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `1.00000` | `1.00000` | `+0.00000` |
| `instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `1.00000` | `0.99750` | `-0.00250` |
| `instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `1.00000` | `1.00000` | `+0.00000` |
| `instructed-deception-gemma-3-27b-it-None` | `1.00000` | `1.00000` | `+0.00000` |
| `instructed-deception-gemma-3-27b-it-s-mo-gemma-3-27b-it` | `1.00000` | `0.99750` | `-0.00250` |
| `varied-deception-Qwen3.5-27B-None` | `0.95750` | `0.97250` | `+0.01500` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.85750` | `0.90750` | `+0.05000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.81125` | `0.72000` | `-0.09125` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `0.87625` | `0.81875` | `-0.05750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `0.98250` | `0.95250` | `-0.03000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.84000` | `0.90250` | `+0.06250` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `0.88250` | `0.79875` | `-0.08375` |
| `varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `0.97000` | `0.95750` | `-0.01250` |
| `varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.93875` | `0.90125` | `-0.03750` |

The original frozen local promotion rule rejected this adapter because it
trailed Phoenix 4.4 validation AUROC. After Phoenix 4.4 subsequently regressed
on the private leaderboard, the team explicitly overrode that local-only
decision and promoted the Qwen-27B soft student as Phoenix 5.0 to test a more
directly transferable supervision path. This is not evidence that the local
comparison reversed.

Independent local-test job `30311515` completed before packaging and reinforced
the caveat: direct margins scored `0.94149` macro AUROC (`0.98427` instructed,
`0.88444` varied), below Phoenix 4.4's `0.95982`. The vector retained much
better resolution than 4.4 at 565 unique scores over 821 rows, with zero missing
label logits and a `12.97s` direct-margin pass. At threshold `0.5`, balanced
accuracy was `0.86786`, recall `0.81429`, and FPR `0.07857`. Phoenix 5.0 is
therefore a private-leaderboard transfer experiment, not a local test winner.

Pure boundary distillation remains much more effective than the earlier
reasoning-SFT-plus-soft-auxiliary formulation (`0.94625` versus `0.93762`). A
future clean experiment could repeat the exact pipeline with the
validation/test-confirmed Qwen3.5-397B teacher rather than adding reasoning or
another loss term.

The separate Qwen3.5-27B thinking diagnostic, job `30310441`, also confirms
that teacher reasoning is unnecessary. Post-reasoning label margins reached
`0.95571` macro AUROC (`0.99792` instructed, `0.89944` varied), slightly below
direct no-thinking `0.95744`, while scoring took `723.3s` instead of `38.2s`.

## Pure Qwen-397B FP8 TVG binary soft distillation (2026-07-28)

Jobs `30342480`--`30342481` repeated the frozen Qwen-27B soft-only recipe with
the completed 2,880-row `Qwen/Qwen3.5-397B-A17B-FP8` Lambda cache. The recipe
remained a fresh rank-16/alpha-32 Qwen3.5-9B LoRA, varied-only data, one epoch,
AdamW `5e-5`, effective batch size 32, and binary soft-target BCE only. There
was no hard-label, completion, pairwise, or reasoning-summary loss.

The teacher checkpoint stored FP8 weights and ran with BF16 activations. vLLM
returned float32 log-probabilities; Python normalized and serialized them at
float64 JSON precision. Training loaded float32 targets, used a BF16 frozen
student base, and retained all 256 trainable LoRA tensors and AdamW state in
float32. The binary margin and BCE were explicitly evaluated in float32.

The locked Transformers environment cannot efficiently train the canonical
multimodal Qwen3.5 class, so this run used the documented text-only
`Qwen3_5ForCausalLM` fallback. The same Slurm job then inserted
`language_model` into all 256 saved tensor paths and verified every tensor
payload after serialization. The migration manifest records the pre/post file
hashes. Evaluation refused legacy paths before vLLM startup.

Training completed 90 optimizer steps in `884.0s` with reported train loss
`4.114`; the complete job took `15m07s`. The direct-margin validation pass had
zero missing scores. As with the Qwen-27B run, the intentionally one-token
generated-output path made all 822 generated strings unparseable; only
`score_metrics.direct_margin` is meaningful.

| adapter | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-397B FP8 soft student | `0.95048` | `0.99813` | `0.88694` | `0.90476` | `0.85714` | `0.04762` | `662/822` |
| Historical Qwen-27B soft student | `0.94625` | `0.99500` | `0.88125` | `0.88810` | `0.83810` | `0.06190` | `544/822` |
| Phoenix 4.4 GRPO reference | **`0.95137`** | **`0.99656`** | **`0.89111`** | `0.89643` | `0.84524` | `0.05238` | `79/822` |

The 397B student improves the historical 27B soft row by `0.00423` macro
AUROC and has substantially more score resolution, but remains `0.00090`
below Phoenix 4.4 on frozen validation. Do not infer that a larger teacher
guarantees promotion or test/private transfer from this small difference.

| validation dataset unit | AUROC |
| --- | ---: |
| `instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-None` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.99250` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.98750` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `1.00000` |
| `instructed-deception-gemma-3-27b-it-None` | `1.00000` |
| `instructed-deception-gemma-3-27b-it-s-mo-gemma-3-27b-it` | `0.99750` |
| `varied-deception-Qwen3.5-27B-None` | `0.98500` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.84750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.70500` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `0.85000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `0.98000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.90000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `0.82000` |
| `varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `0.95500` |
| `varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.94000` |

The mandatory same-session vLLM activation fingerprint also passed. Adapter
and base float32 score digests differed; only 12/822 scores were exactly equal,
mean absolute difference was `0.08141`, and maximum absolute difference was
`0.49463`. This rules out the silent-base failure for this evaluation.

The canonical adapter was promoted as Phoenix 6.3 and published at
`Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16`, revision
`59a9c57982c8430ff9e8bb64c3e53b8fb775dbc2`. The remote LFS weight digest
matches the local validated file:
`fac7b031bf52bc5597ee02a6dec20f355b3c3297bf110d3bb0233f629a69004d`.

## Qwen-397B soft-distillation hyperparameter sweep (2026-07-29)

A Lambda `gpu_1x_h100_sxm5` campaign optimized the frozen Qwen-397B
identity-probability BCE recipe while keeping the submitted model fixed at
`Qwen/Qwen3.5-9B`. The rank-16 stage crossed learning rates `1e-5`, `2e-5`,
`5e-5`, and `1e-4` with `0.5`, `1`, and `2` epochs. After selecting the
rank-16 response surface, rank 24 tested all epoch lengths at `5e-5` and all
four learning rates at two epochs. Every arm retained alpha 32, the varied-only
2,880-row cache, effective batch size 32, and direct normalized literal `0|1`
validation margins. Selection used only the 822-row validation split; no local
test evaluation was spent.

| rank | learning rate | 0.5 epoch | 1 epoch | 2 epochs |
| ---: | ---: | ---: | ---: | ---: |
| 16 | `1e-5` | `0.94768` | `0.94786` | `0.95107` |
| 16 | `2e-5` | `0.94798` | `0.94935` | `0.95179` |
| 16 | `5e-5` | `0.95071` | `0.95006` | **`0.95393`** |
| 16 | `1e-4` | `0.94673` | `0.94887` | `0.95083` |
| 24 | `1e-5` | — | — | `0.95131` |
| 24 | `2e-5` | — | — | **`0.95369`** |
| 24 | `5e-5` | `0.94887` | `0.94923` | `0.95339` |
| 24 | `1e-4` | — | — | `0.95256` |

The winner is
`qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1`: rank 16, alpha 32, AdamW
`5e-5`, and two epochs. It scored `0.95393` validation macro AUROC,
`0.99833` instructed AUROC, and `0.89472` varied AUROC. Its secondary
threshold-0.5 balanced accuracy was `0.90595`, with `0.85238` recall and
`0.04048` FPR. This is `+0.00345` over the historical one-epoch Qwen-397B
student (`0.95048`) and `+0.00018` over the corrected Qwen-27B/Phoenix 6.2
adapter-prompt reference (`0.95375`). The best rank-24 arm, `2e-5` for two
epochs, reached `0.95369`, losing to rank 16 by `0.00024`; do not promote rank
24 from this sweep.

The winner retained 665 unique float scores over 822 rows. There were 157
duplicate-score excess rows and 281 rows participating in a tie, so the output
is meaningfully continuous but not tie-free. All direct label margins were
present. The deliberately one-token generation probe did not emit complete
binary strings and is not an alternate score path. The same-session activation
fingerprint passed: only 4/822 adapter and base margins were exactly equal and
their mean absolute difference was `0.10250`.

| validation dataset unit | AUROC |
| --- | ---: |
| `instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-None` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.98750` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.99500` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `1.00000` |
| `instructed-deception-gemma-3-27b-it-None` | `1.00000` |
| `instructed-deception-gemma-3-27b-it-s-mo-gemma-3-27b-it` | `0.99750` |
| `varied-deception-Qwen3.5-27B-None` | `0.98750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.83000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.72250` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `0.86750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `0.98750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.90000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `0.85500` |
| `varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `0.96000` |
| `varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.94250` |

The SXM5 throughput study also changed the execution recipe. With effective
batch size 32, the fallback attention path improved from 1.798 samples/s at
microbatch 2 to 4.279 samples/s at microbatch 16; microbatch 32 was slower and
used roughly 80.98 GB. Trainer-level `torch.compile` spent 5m22s without
finishing its first step and was rejected. Isolated pinned FLA 0.5.2 packages
instead reached 8.811 samples/s at microbatch 8 and 8.791 at microbatch 16.
The sweep therefore used microbatch 8 with four-step accumulation, FLA, and no
Trainer compilation. Full runs stabilized around 9.84--10.44 samples/s while
leaving materially more HBM headroom than microbatch 32.

The selected canonical adapter file has SHA-256
`da32c67a2ea0d9834fb90b93dac84a9ee68f05b2a2023b8ea272704fdb58c40d`.
All 256 tensors use canonical Qwen3.5 PEFT paths and no legacy tensor path
remains. Retain this rank-16/two-epoch arm as the validation-selected
Qwen-397B distillation result. Any test or private-leaderboard promotion should
be a frozen follow-up, not another validation retune.

The same optimized recipe was then applied to the regular-OpenRouter 397B
explicit-digit cache. It reached `0.95173` validation macro AUROC (`0.99500`
instructed, `0.89403` varied), improving its matched one-epoch predecessor by
`0.00202` but losing to the original-prompt optimized adapter by `0.00220`.
It produced 787 unique scores over 822 rows, 35 duplicate-score excess rows,
and zero missing margins. Keep it as a completed prompt-transfer ablation; do
not replace the original-prompt winner.

The original-prompt winner is now the bundled Phoenix 6.3 main adapter and is
published at
`Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2`, revision
`11cd26e44b77cf86064ed2246504952681d25695`. The remote LFS SHA-256 equals the
validated local digest above. The notebook retains its existing filename,
restores the exact original training rubric, and was not submitted while the
competition queue remained active.

The user then authorized the first and only frozen local-test evaluation of
this validation-selected adapter. The matched vLLM path used the exact original
prompt, excluded source reasoning, and scored normalized literal next-token
`0|1` logits directly after `Prediction:`. All 821 rows were scored.

| split | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation, matched vLLM | `0.95369` | `0.99792` | `0.89472` | `0.90833` | `0.85476` | `0.03810` | `768/822` |
| frozen local test | **`0.95869`** | `0.99542` | **`0.90972`** | `0.88690` | `0.82381` | `0.05000` | `774/821` |

Test macro AUROC is `+0.00500` over the matched vLLM validation reference
(`+0.00476` over the canonical `0.95393` result). It is `0.00113` below the
Phoenix 4.4 GRPO test reference (`0.95982`) and `0.00161` below the separate
regular-OpenRouter/explicit-prompt student (`0.96030`). The direct 397B teacher
test ceiling remains higher at `0.9654`; that teacher used the regular
OpenRouter checkpoint rather than the FP8 cache used to train this student.
These small cross-method differences do not justify test-set retuning.

The threshold-0.5 metrics above remain the frozen binary diagnostics. A
test-only threshold sweep peaked at `0.91548` balanced accuracy at
`0.13296423`, but this is diagnostic and must not replace the validation-frozen
submission threshold. The test vector has 47 duplicate-score excess rows and
84 rows participating in ties. One-token scoring took `8.54s`
(`96.11` rows/s), excluding model startup. The exact preserved predictions CSV
has SHA-256
`4323c98bf8a6fa67211d8e3dfcbceaf2fde5d2ec10539a365176973a15b5d1cf`.

| test dataset unit | AUROC |
| --- | ---: |
| `instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-None` | `0.99000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.97750` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.99750` |
| `instructed-deception-gemma-3-27b-it-None` | `0.98000` |
| `instructed-deception-gemma-3-27b-it-s-mo-gemma-3-27b-it` | `1.00000` |
| `varied-deception-Qwen3.5-27B-None` | `0.95500` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.85750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.86750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `0.87000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `0.94250` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.97250` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `0.88250` |
| `varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `0.89750` |
| `varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.94250` |

Ignored reproducibility artifacts:

- `results/blackbox/qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1/test_vllm_logits/`

## Qwen-397B soft-objective ablation (2026-07-28)

Jobs `30342642`--`30342645` tested whether the pure Qwen-397B soft student lost
AUROC because its binary teacher probabilities were too saturated. Binary
soft-target BCE is forward teacher-to-student KL up to the teacher entropy
constant, so this was an output-geometry ablation rather than a nominal
BCE-versus-KL comparison.

The frozen 2,880-row teacher cache had label-blind margin mean
`-2.2910156273` and population standard deviation `6.3133151107`. Its raw
probabilities spanned `0.0000147--0.999877`; 2,208/2,880 were outside
`[0.01, 0.99]`. Both new arms transformed the teacher target to
`(margin - mean) / std`, spanning approximately `-1.399--1.788`. One arm
applied BCE to the corresponding `0.198--0.857` probability, while the other
used Smooth L1 margin regression with delta `1.0`. Everything else remained
matched to the pure soft-only recipe: a fresh rank-16/alpha-32 Qwen3.5-9B
LoRA, varied-only data, one epoch, AdamW `5e-5`, effective batch size 32, and
no hard-label, completion, pairwise, or reasoning loss.

Both students completed 90 steps and canonical PEFT migration in 15m48s.
Standardized BCE reported train loss `9.931`; standardized Huber reported
`2.351`. These losses are not comparable across objectives. Both validation
jobs had zero missing direct margins and passed the same-session LoRA-effect
fingerprint.

| objective | macro AUROC | delta | instructed | varied | BA at 0.5 | recall | FPR | unique scores | cross-label ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity BCE / forward KL | **`0.950476`** | — | **`0.998125`** | `0.886944` | **`0.904762`** | `0.857143` | **`0.047619`** | **662/822** | **0** |
| Standardized-target BCE | `0.947381` | `-0.003095` | `0.996667` | `0.881667` | `0.892857` | **`0.885714`** | `0.100000` | 106/822 | 50 |
| Standardized-margin Huber | `0.949940` | `-0.000536` | `0.995208` | **`0.889583`** | `0.890476` | **`0.885714`** | `0.104762` | 105/822 | 51 |

The transformed arms changed score scale materially but retained broadly
similar ordering: Spearman correlation with the identity-BCE score was
`0.9783` for standardized BCE and `0.9744` for Huber. Their compressed target
range also compressed BF16 student margins, collapsing score resolution and
introducing cross-label ties where the identity objective had none. Huber's
small varied gain was offset by a larger instructed loss under the
21-unit macro average.

| validation dataset unit | identity BCE | standardized BCE | standardized Huber |
| --- | ---: | ---: | ---: |
| `instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-None` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | **`0.99250`** | `0.97750` | `0.97375` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | **`1.00000`** | `0.99750` | `0.99625` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.98750` | **`0.99000`** | **`0.99000`** |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-gemma-3-27b-it-None` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-gemma-3-27b-it-s-mo-gemma-3-27b-it` | **`0.99750`** | `0.99500` | `0.98250` |
| `varied-deception-Qwen3.5-27B-None` | **`0.98500`** | `0.97750` | `0.97625` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | **`0.84750`** | `0.82625` | `0.84000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.70500` | `0.73250` | **`0.75125`** |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | **`0.85000`** | `0.83625` | `0.83250` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | **`0.98000`** | `0.95750` | `0.97250` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.90000` | `0.89500` | **`0.91875`** |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `0.82000` | `0.82625` | **`0.83875`** |
| `varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `0.95500` | `0.95875` | **`0.96250`** |
| `varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | **`0.94000`** | `0.92500` | `0.91375` |

Decision: retain the identity-probability BCE student used by Phoenix 6.3.
Neither standardized loss improves validation macro AUROC, and the score-tie
regression is directly adverse to the continuous leaderboard metric. Do not
spend a local-test evaluation, replace the published adapter, or select raw
Huber/temperature variants post hoc from this result.

## Qwen-397B soft-target DataRater filtering (2026-07-29)

The earlier DataRater screen used hard-label meta gradients and
reasoning-summary completion gradients, so it did not test whether dense
Qwen-397B feedback makes gradient-based filtering more useful. This follow-up
used the exact identity-probability BCE objective for both sides: the meta
gradient and every candidate gradient scored the normalized literal `0|1`
logits directly against the cached Qwen-397B soft target. Ground-truth labels
were used only to preserve every dataset/label stratum.

The rank-16 scoring probe covered the attention and MLP projections in the last
Qwen3.5-9B block, or 1,277,952 trainable parameters. A 36-meta-row,
72-candidate calibration compared exact per-example gradient dots with
finite-difference scores. Jobs `30342926`--`30342929` produced:

| finite-difference epsilon | Pearson vs exact | Spearman vs exact | sign agreement | top-half overlap |
| ---: | ---: | ---: | ---: | ---: |
| `0.01` | `0.10666` | `0.26482` | `0.23611` | `17/36` |
| `0.03` | `0.86527` | `0.85120` | `0.80556` | `31/36` |
| `0.10` | **`0.98666`** | **`0.99183`** | **`1.00000`** | **`34/36`** |

The full scorer therefore froze epsilon `0.10`. Job `30342972` used 144
dataset/label-stratified meta rows and scored the remaining 2,736 candidates
in 24m52s. Each 50% manifest contained 1,368 rows, exactly 76 from each of 18
dataset/label strata. Dot and loss had only weak score correlation (Pearson
`0.1389`, Spearman `0.2094`) and selected materially different data:

| selection | rows | mean gradient dot | mean soft BCE | positive-dot fraction |
| --- | ---: | ---: | ---: | ---: |
| All candidates | 2,736 | `+0.03924` | `0.31895` | `0.6173` |
| Matched random | 1,368 | `+0.03694` | `0.33579` | `0.6060` |
| High loss | 1,368 | `+0.06968` | `0.56585` | `0.6235` |
| Gradient dot | 1,368 | `+0.24447` | `0.38710` | `0.7566` |

Dot/loss shared 856 rows (Jaccard `0.4553`), dot/random shared 690
(`0.3372`), and loss/random shared 698 (`0.3425`). Thus a negative downstream
result cannot be attributed to the dot selector degenerating to either
control.

All students used a fresh rank-16/alpha-32 LoRA, AdamW `5e-5`, effective batch
size 32, and exactly 90 optimizer steps. The first training jobs
`30342973`--`30342975` exposed an Inductor `CantSplit` failure in the
Qwen3.5 hybrid-layer backward graph before optimizer step one. Eager-mode
one-step smoke `30343584` passed in 11.3s. Frozen reruns `30343636`--`30343638`
therefore disabled the model's internal Dynamo compilation and completed in
15m31s, 16m52s, and 15m37s. This changes execution only; the loss, initialization,
data order, updates, and random control remain matched. Reported train losses
were `3.890` random, `6.132` high-loss, and `2.481` gradient-dot.

Shared vLLM jobs `30343639` and `30343641` loaded the full-data baseline,
random, high-loss, and dot adapters in opposite orders. Every one-token
generated string was intentionally unparseable, while the direct scorer had
zero missing label margins. The table averages the two adapter positions:

| training data | macro AUROC | delta vs baseline | instructed | varied | BA at 0.5 | recall | FPR | unique scores fwd/rev | within-unit cross-label ties fwd/rev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full varied baseline | `0.949673` | — | **`0.998073`** | `0.885139` | `0.903571` | **`0.854762`** | `0.047619` | `658/655` | `3/2` |
| Random keep 50% | `0.950982` | `+0.001310` | `0.996771` | `0.889931` | `0.901786` | `0.853571` | `0.050000` | `669/673` | `2/0` |
| High-loss keep 50% | **`0.951845`** | **`+0.002173`** | **`0.998125`** | **`0.890139`** | **`0.907738`** | `0.844048` | `0.028571` | `676/646` | `0/2` |
| Gradient-dot keep 50% | `0.943363` | `-0.006310` | `0.995313` | `0.874097` | `0.809524` | `0.623810` | **`0.004762`** | `618/632` | `2/1` |

Forward/reverse score Spearman correlations were `0.99954` baseline,
`0.99952` random, `0.99837` loss, and `0.99918` dot. The corresponding macro
AUROCs moved by `0.00101`, `0.00054`, `0.00036`, and `0.00101`. The original
single-session baseline was `0.950476`; its new position average is
`0.949673`, an additional warning against promoting the high-loss arm's small
nominal gain.

| validation dataset unit | full baseline | random 50% | high loss 50% | gradient dot 50% |
| --- | ---: | ---: | ---: | ---: |
| `instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None` | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-None` | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.99250` | `0.99250` | **`0.99500`** | `0.98875` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | **`1.00000`** | `0.99250` | `0.99750` | `0.99500` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | **`0.98750`** | `0.98000` | **`0.98750`** | `0.96750` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.99937` | **`1.00000`** | **`1.00000`** | **`1.00000`** |
| `instructed-deception-gemma-3-27b-it-None` | `1.00000` | `1.00000` | `1.00000` | `1.00000` |
| `instructed-deception-gemma-3-27b-it-s-mo-gemma-3-27b-it` | **`0.99750`** | `0.99625` | **`0.99750`** | `0.99250` |
| `varied-deception-Qwen3.5-27B-None` | **`0.98438`** | `0.97625` | `0.97937` | `0.98063` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.84375` | **`0.87250`** | `0.80000` | `0.78750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.69625` | `0.72375` | **`0.75188`** | `0.72875` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `0.85375` | `0.85000` | **`0.87500`** | `0.84000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | **`0.98000`** | `0.97250` | **`0.98000`** | `0.96500` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.89500` | `0.91000` | `0.90750` | **`0.91875`** |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `0.82000` | `0.81500` | **`0.83875`** | `0.78500` |
| `varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `0.95375` | `0.94500` | `0.93875` | **`0.95625`** |
| `varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.93938` | **`0.94500`** | `0.94000` | `0.90500` |

The dot selector's failure is not only threshold calibration. It lowers varied
AUROC by `0.01104` against the matched baseline. Selection also makes the
ground-truth-positive soft targets strongly asymmetric: their mean teacher
probability is `0.7383` under random selection, `0.6625` under high loss, and
only `0.5205` under gradient-dot selection. The fraction on the correct side
of 0.5 is respectively `74.85%`, `66.96%`, and `51.17%`; for selected dot
negatives it is `99.27%`. This concentrates ambiguous or teacher-conflicting
positive targets, pulling validation positive scores down and cutting recall.

Decision: retain the full identity-BCE student used by Phoenix 6.3. The
gradient-dot selector is worse than both controls, while high-loss filtering
misses the frozen `+0.005` validation promotion rule and beats matched random
by only `0.00086`. Do not spend a local-test evaluation, promote either
filtered adapter, or extend this screen to other keep fractions. Dense
feedback does not rescue this initialization-state, last-block DataRater
approximation on the already curated Qwen-only varied cache.

## Kimi K3 two-token soft distillation (2026-07-29)

Fireworks exposed literal next-token log probabilities for
`moonshotai/kimi-k3` through OpenRouter, allowing a compact teacher target
without generated reasoning. The frozen no-thinking Truth Value Guard prompt
requested `Prediction:`, and the cache retained the normalized log-odds over
literal `0|1`. The complete 2,880-row varied-deception train cache had zero
missing logits, cost `$3.61396`, and scored macro AUROC `0.95072`. Kimi exceeded
the matched Qwen3.5-397B FP8 teacher by `0.03094` AUROC on these rows; their
probabilities still correlated strongly (Pearson `0.93642`).

The student retained the matched rank-16/alpha-32, AdamW `5e-5`, effective
batch-size-32 recipe but used two epochs of binary soft-target BCE. Training on
one rented H100 SXM5 took 180 optimizer steps. The saved canonical multimodal
PEFT adapter has 256 FP32 tensors and weight SHA-256
`c5025a39dd05af16405c692a0c1b70657afd4f8e4a4e634bd789b5d67b4a9eb0`.

Frozen validation used the exact direct `Prediction:` margin and no generated
judge reasoning:

| adapter | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen397 soft student | `0.95393` | `0.99833` | `0.89472` | `0.90595` | — | — | `665/822` |
| **Kimi K3 soft student** | **`0.95994`** | **`0.99833`** | **`0.90875`** | **`0.91548`** | `0.86429` | `0.03333` | `765/822` |

The gain is `+0.00601` macro AUROC and clears the frozen `+0.005` promotion
bar. All 822 rows had label logits. A base-model fingerprint confirmed that
the LoRA changed 815/822 scores (mean absolute change `0.11062`, maximum
`0.74526`).

The canonical adapter is published at
`Jazhyc/aletheias-phoenix-v7-kimi-k3-tvg-soft-r16-ep2`, revision
`74e5cbdf4d3c93c57f4b304c0d66cb1c46c84e92`. Phoenix 7.0 keeps its filename,
renderer, prompts, route precedence, batching, and rank-1 intent blend; only
the shared primary-route adapter changes. Treat this as a frozen private
transfer candidate: the training cache is still varied-only and therefore
implicitly Qwen-family-only.

### Kimi student learning-rate and epoch sweep

A matched H100 sweep tested whether the Qwen397-selected `5e-5`, two-epoch
recipe transferred to Kimi's materially softer target distribution. Kimi's
mean binary-target entropy was `0.1074` nats versus `0.0769` for Qwen397, and
`64.65%` versus `76.67%` of probabilities lay outside `[0.01, 0.99]`.
Rank 16, alpha 32, effective batch size 32, all 2,880 cached rows, and the
direct-margin evaluator remained fixed.

| learning rate | 0.5 epoch | 1 epoch | 2 epochs |
| ---: | ---: | ---: | ---: |
| `1e-5` | `0.94708` | `0.94827` | `0.95185` |
| `2e-5` | `0.94816` | `0.95143` | `0.95405` |
| `5e-5` | `0.95042` | `0.95274` | `0.96042` |
| `1e-4` | `0.95054` | `0.95702` | **`0.96071`** |

The nominal grid winner was `1e-4` for two epochs, but its matched-session
advantage over the submitted `5e-5`, two-epoch adapter was only `0.00030`
macro AUROC. This is far below the frozen `+0.005` promotion bar and below
previously observed session/position sensitivity. Do not replace the submitted
adapter from this difference.

Because both top learning rates still peaked at the two-epoch boundary, fresh
three- and four-epoch runs were trained at `1e-4` and rescored together with
the two-epoch winner:

| epochs | macro AUROC | instructed | varied | BA at 0.5 | unique |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **2** | **`0.96179`** | **`0.99667`** | `0.91528` | `0.90714` | `752/822` |
| 3 | `0.96101` | `0.99521` | **`0.91542`** | **`0.92024`** | `746/822` |
| 4 | `0.95887` | `0.99479` | `0.91097` | `0.90476` | `763/822` |

Three epochs gained only `0.00014` varied AUROC while losing `0.00077` macro
AUROC; four epochs regressed materially. Two epochs is therefore the validated
horizon. Retain the already published/submitted `5e-5`, two-epoch checkpoint:
the learning-rate distinction is unresolved noise, while the evidence against
training beyond two epochs is consistent.

### Frozen Kimi student local-test confirmation

After closing the validation sweep, the already submitted `5e-5`, two-epoch
adapter received its first and only full 821-row local-test evaluation. The
test used the exact frozen ordinary Phoenix prompt, excluded source reasoning,
generated no judge reasoning, and scored normalized literal `0|1` logits
directly after `Prediction:`.

| split | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `0.95994` | `0.99833` | `0.90875` | `0.91548` | `0.86429` | `0.03333` | `765/822` |
| **Frozen local test** | **`0.96327`** | `0.99229` | **`0.92458`** | `0.88571` | `0.81905` | `0.04762` | `771/821` |

Test macro AUROC increased by `0.00333` and varied AUROC by `0.01583`, while
instructed AUROC fell by `0.00604`. Relative to the matched Qwen397 soft
student test row (`0.95869/0.99542/0.90972`), the Kimi student gains `0.00458`
macro and `0.01486` varied AUROC but loses `0.00313` instructed AUROC.

The Kimi teacher remains a materially higher ceiling: its direct test result
was `0.98244` macro (`0.99865` instructed, `0.96083` varied), leaving student
gaps of `0.01917`, `0.00635`, and `0.03625`. Do not claim that the compact
student matches or exceeds its teacher.

| test dataset unit | AUROC |
| --- | ---: |
| `instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None` | `0.95000` |
| `instructed-deception-Qwen3.5-27B-None` | `0.99000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.98250` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `1.00000` |
| `instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `1.00000` |
| `instructed-deception-gemma-3-27b-it-None` | `0.98500` |
| `instructed-deception-gemma-3-27b-it-s-mo-gemma-3-27b-it` | `1.00000` |
| `varied-deception-Qwen3.5-27B-None` | `0.96250` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1` | `0.89750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3` | `0.87750` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4` | `0.88875` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5` | `0.94000` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6` | `0.98125` |
| `varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7` | `0.87750` |
| `varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b` | `0.91125` |
| `varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b` | `0.98500` |

All direct margins were present. Scores had 771 unique values, 50
duplicate-score excess rows, and 90 rows participating in a tie. Direct
adapter scoring took `8.34s` (`98.4` rows/s), excluding model startup. The
same-session LoRA fingerprint passed: only 4/821 adapter and base scores were
exactly equal, with mean absolute difference `0.11643` and maximum `0.94138`.
The preserved generations JSONL SHA-256 is
`56a1a8ba94d9d24309337cd69618c937253312b632d2efa9e4453d388de9f99a`.

## Optimized Qwen-397B reasoning-GRPO continuation (2026-07-29)

The validation-selected original-prompt soft student
`qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1` initialized a new rank-16
varied-only GRPO run. GRPO received only observable output text and an explicit
instruction to produce its own concise `<reasoning>...</reasoning>` analysis
before `Prediction:0|1`; source assistant reasoning remained excluded.
Promotion scoring used the exact original binary-only student/Phoenix prompt
and one constrained vLLM next-token pass.

H100 speed probes held batch 4 and four generations fixed. Generation batch 32
needed `139.7s` for 16 warm steps, versus `214.8s` at generation batch 64.
The selected `0.35` vLLM memory fraction was the smallest tested viable setting
on the 80 GB H100; `0.25` left no KV-cache blocks. The full one-epoch run
completed 360 steps in `2863.9s`. Its adapter SHA-256 is
`f5098fd8da0642dd8e5ff5f5bf63a9a1bddce2ecad050d3f894425d128e82384`.
Reasoning shortened from roughly 140 tokens early to 60 tokens in the final
logged batch, while clipping fell from about 14% to zero.

| adapter | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Optimized SFT, matched vLLM | **`0.95369`** | **`0.99792`** | **`0.89472`** | **`0.90833`** | **`0.85476`** | **`0.03810`** | **`768/822`** |
| Reasoning GRPO epoch 1 | `0.94500` | `0.99385` | `0.87986` | `0.88452` | `0.83333` | `0.06429` | `582/822` |

The GRPO score loses `0.00869` macro AUROC and `0.01486` varied AUROC. Cleaner
reasoning and lower training loss therefore do not imply a better direct
ranking. The initial continuation fails the conditional gate for further
learning-rate/epoch tuning: preserve it remotely as a negative result, but do
not test it, upload it, or replace the Phoenix 6.3 SFT adapter.

## Leaderboard-aligned GRPO continuation logits (2026-07-27)

An inference-input audit found that the saved GRPO configurations retained
`judge.include_reasoning=true`. This supplied the dataset's source
`assistant_reasoning` field to both the early base-Qwen GRPO adapter and the
later privileged-distillation continuations. Because leaderboard datasets do
not expose that field, the earlier roughly `0.965` test AUROC is not a valid
leaderboard estimate.

The logits evaluator now supports `--exclude-reasoning`. The corrected path
renders only the observable prompt and assistant answer, appends
`<reasoning>\n</reasoning>\nPrediction:`, and scores the normalized next-token
margin over literal `0` and `1`. It performs one Qwen3.5-9B forward pass per
row; it does not generate reasoning at inference.

Frozen validation job `30304158` compared the one- and two-epoch varied-only
GRPO continuations from the selected rank-16 privileged-distillation adapter:

| adapter | macro AUROC | instructed | varied | BA at 0.5 | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phoenix/PID SFT direct reference | `0.93815` | `0.99354` | `0.86431` | — | — |
| GRPO epoch 1 | **`0.95137`** | **`0.99656`** | `0.89111` | **`0.89643`** | `79/822` |
| GRPO epoch 2 | `0.94935` | `0.98667` | **`0.89958`** | `0.86905` | `69/822` |

Macro AUROC selected epoch 1 before local-test evaluation. Its frozen
validation threshold was `0.37754068`, where validation BA was `0.91071`.
Epoch 2 was rejected despite its higher varied AUROC because it lost overall
and instructed ranking.

Frozen local-test job `30304374` then evaluated epoch 1 exactly once:

| adapter | macro AUROC | instructed | varied | BA at 0.5 | BA at frozen threshold | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phoenix/PID SFT direct reference | `0.93423` | `0.98177` | `0.87083` | — | — | — |
| GRPO epoch 1 | **`0.95982`** | **`0.99073`** | **`0.91861`** | `0.88690` | `0.89881` | `79/821` |

The test run had recall `0.82857` and FPR `0.05476` at threshold `0.5`; the
validation-frozen threshold gave recall `0.85714` and FPR `0.05952`. Scoring
took `38.85s` for 821 rows. The generic result artifact also reports a
test-oracle threshold, but that value was inspected only as a diagnostic and
must not be used for selection or deployment.

This is the strongest tracked leaderboard-aligned local-test judge by macro
AUROC, narrowly exceeding the much more expensive Qwen3.5-27B three-generation
D/K/S ensemble (`0.9573`). Promote the epoch-1 adapter as the next submission
candidate. Retain the continuous logit score despite its 742 tied rows, keep
the hard `deceptive` threshold separate, and do not re-enable source reasoning
or reasoning generation at inference.

## Full-data blind GPT-OSS reasoning SFT (2026-07-27)

Jobs `30313949`--`30313951` completed the ground-truth-blind control requested
to distinguish privileged-information supervision from reasoning-target SFT.
The teacher received all 2,880 varied-deception training rows, but never saw a
label or answer-key placeholder. GPT-OSS-120B generated a concise
material-claim rationale and its own `Prediction:0|1`. The student used the
same default one-epoch rank-16/alpha-32 Qwen3.5-9B AdamW `5e-5` recipe with
effective batch size 32.

Five teacher outputs were unparseable, leaving 2,875 targets. Of those, 415
teacher predictions disagreed with ground truth. They were retained without
correction or label-based filtering. The teacher's valid-target macro train BA
was only `0.85564`, with substantial unit variation (`0.67476--0.95625`).
Training completed 90 steps in `979.9s` with reported loss `6.862`. The adapter
weight SHA-256 is
`bfe4a35ceb11055e7b7e77d29cc7027ccb5daa355703d9acd4d14eb03db01931`.

Validation generation and direct-label scoring gave:

| readout | macro AUROC/BA | instructed | varied | unique scores | missing/parse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generated binary | `0.90595` | `0.97708` | `0.81111` | `2/822` | `2` parse |
| Direct `0|1` margin | `0.93762` | `0.99344` | `0.86319` | `347/822` | `0` missing |
| Privileged-summary direct reference | `0.93815` | `0.99354` | `0.86431` | `348/822` | `0` missing |

The blind and privileged direct score vectors have Pearson correlation
`0.99940`; mean absolute score difference is `0.00611`. Their AUROC difference
is only `-0.00054` overall and is not a meaningful validation separation. The
blind student also improves generated binary BA over the privileged student's
original `0.90000`, despite learning from the teacher's noisy decisions.

This is strong local evidence that the earlier advantage over prediction-only
SFT mostly comes from learning a compact factual-judgment reasoning procedure
or output manifold, not from privileged labels embedded in post-hoc rationales.
It does not prove privileged information is useless OOD: all training varied
rows are also Qwen-family rows, and private transfer may still differ. Phoenix
5.1 therefore promotes this adapter as an explicit private-leaderboard control,
not as a validation winner over the privileged adapter.

Frozen local-test job `30314450` confirmed the same conclusion. Direct margins
scored `0.93452` macro AUROC (`0.98073` instructed, `0.87292` varied), with 345
unique values over 821 rows, zero missing label logits, and `14.98s` scoring
time. The privileged-summary reference was `0.93423`, so the blind student is
fractionally higher but remains an empirical tie. Generated binary BA was
`0.91071`; its eight parse errors do not affect the direct-margin result. The
validation-frozen `0.19` threshold gave `0.86786` macro test BA.
