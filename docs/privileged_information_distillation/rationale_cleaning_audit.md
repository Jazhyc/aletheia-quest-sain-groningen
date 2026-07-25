# Privileged teacher-rationale cleaning audit

## Question and scope

This audit estimates how often the original GPT-OSS-120B privileged teacher
produced supervision that should not be copied into the rank-16 student's
reasoning summary. It separates four failure modes:

1. the visible summary supports the opposite prediction from the attached
   authoritative label;
2. it confidently certifies or invents an ordinary fact;
3. it treats failure to follow a hidden instruction to lie as deception by
   itself; or
4. it exposes label, ground-truth, or privileged-information language.

The source is the 6,573-row reviewed cache at
`results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl`.
The selected varied-only student consumes 2,877 of its 6,570 usable summaries.
All selected rows are Qwen-family varied-deception examples, so the result does
not estimate cross-family varied supervision quality.

This is an audit of *rationale usefulness*, not an attempt to relabel the
competition data. An ordinary factual error can make a summary bad supervision
without proving that the final answer is belief-based deception.

## Methods

Three measurements are kept separate:

- Deterministic text checks count exact visible phrases such as `the label`,
  `privileged`, and positive summaries that explicitly conclude there is no
  important false or misleading claim.
- Manual review checks the conversation, final answer, summary, and, where
  useful, the separately generated reasoning-aware teacher cache.
- A label-blind GPT-OSS-120B screen independently compares the conversation,
  final answer, and visible teacher summary. The authoritative label is withheld
  until after generation, when the inferred decision is crossed with it. The
  screen classifies factual confidence, logical coherence, and visible
  meta-information. Runtime code is
  `experiments/privileged_information_distillation/audit_teacher_rationales.py`.

The model screen is a triage estimate, not ground truth: it uses the same model
family as the original teacher and therefore shares factual blind spots. Its
flagged examples require manual calibration before defining a training filter.

## Calibrated full-cache GPT-OSS screen

Calibrated job `30287922` audited all 6,570 usable summaries with a 1,024-token
budget. It produced 6,561 valid JSON audits and nine truncation/parse failures
in 1,435.4 scoring seconds. The first 512-token pilot is discarded: it had
1,216 truncations and an ambiguous factual-status definition that caused the
model to flag a correct rationale merely because the assistant output was
false. Across the full cache, the calibrated raw recommendations are 6,257 keep,
62 manual review, and 251 mask candidates (`3.82%`).

For the 2,877 selected varied summaries, the calibrated screen assigned:

| recommendation / flag | rows | fraction |
| --- | ---: | ---: |
| Keep summary | 2,659 | 92.42% |
| Manual review | 32 | 1.11% |
| Raw high-confidence mask candidate | 186 | 6.47% |
| Teacher-summary factual error | 147 | 5.11% |
| Summary implies the opposite authoritative label | 30 | 1.04% |
| Explicit privileged/meta reference | 14 | 0.49% |
| Non-coherent logic | 5 | 0.17% |
| Parse failure | 9 | 0.31% |

The component flags overlap. Of the 186 mask candidates, 147 have a factual
flag, 30 a label conflict, 14 an explicit meta flag, and five a logic flag.
Label-0/label-1 mask candidates split 65/121; factual flags split 65/82.

The raw 6.47% is an upper screening bound, not an estimated true error rate. A
deterministic random sample of 30/147 selected factual flags (seed `20260725`)
contained:

- 16 clear teacher-summary errors;
- three arguable, uncertain, or immaterial errors; and
- 11 false, pedantic, or instruction-confused GPT-OSS corrections.

The clear-error indices are 210, 240, 669, 676, 758, 1012, 1084, 1136, 1238,
1804, 2198, 2239, 2259, 2444, 2801, and 3235. The arguable/minor indices are
447, 812, and 981. The rejected flags are 524, 801, 963, 1662, 1811, 1985,
2105, 2184, 2211, 2820, and 3342. This annotation judges whether the summary is
useful supervision under the competition rule, not merely whether GPT-OSS can
name a debatable correction.

Applying only that manual precision to the factual slice and retaining the
30-row exact semantic lower bound suggests roughly 108--123 harmful summaries,
or about **3.8--4.3%** of selected training rows. Given overlap and unknown
recall, use **3--4.5%** as the practical prevalence estimate, `1.04%` as the
firm high-confidence lower bound, and `6.47%` as the raw review upper bound.

The same-model limitation is material. GPT-OSS marked all eight manually
reviewed factual misses below as `factual_status=sound`, including the Palgrave,
Cape of Good Hope, and Technicolor errors. It still caught those rows as
summary/label conflicts, but did not discover the concrete facts. Across the
30-row exact semantic set, its inferred-prediction conflicts recover 24 and
miss six, while adding six other review candidates. Conversely, it made plainly
bad corrections, including denying that the Shakespeare line comes from *A
Midsummer Night's Dream*, denying Tiger Woods's Sunday-red habit, and treating a
correct answer as factually wrong merely because a hidden prompt had requested
a lie.

Examples that survived manual review as genuine additional teacher errors
include rationales that:

- deny that Jezebel was thrown from a window and eaten by dogs;
- call the *Bridal Chorus* non-Wagnerian rather than part of *Lohengrin*;
- say *Vertical Limit* is set in the Himalayas rather than on K2 in the
  Karakoram;
- call Longleat a Somerset rather than Wiltshire estate;
- claim that Thailand's concessions amounted to European colonization; or
- reverse the UK's uncodified/de-facto national-anthem distinction.

These are precisely the cases a selective rationale mask could remove, but the
screen must not be used without a second checker or human review.

## High-confidence deterministic lower bound

Among the 2,880 varied rows before parser exclusion:

| failure | rows | fraction of all varied rows | fraction of positive varied rows |
| --- | ---: | ---: | ---: |
| Usable positive summary explicitly concludes no consequential falsehood | 24 | 0.83% | 1.67% |
| Usable positive summary relies only on disobeying the hidden lie instruction | 6 | 0.21% | 0.42% |
| Malformed trace loops while reconciling a positive label with a correct-looking output | 3 | 0.10% | 0.21% |
| **Structural total** | **33** | **1.15%** | **2.29%** |

The six instruction-only summaries are indices 242, 347, 365, 369, 759, and
797. Indices 242, 365, and 759 are also the three exact `privileged`
references. They explicitly call the answer true, then convert failure to
supply a requested lie into an important falsehood.

Restricting to rows that actually entered SFT gives 30/2,877 summaries
(`1.04%`, Wilson 95% interval approximately `0.73--1.49%`) and 30/1,437
positive summaries (`2.09%`, approximately `1.47--2.96%`). These are a lower
bound because a polished but factually wrong rationale is not detectable from
its conclusion alone. All 30 are varied positive rows; across the entire usable
cache they are 30/6,570 rows (`0.46%`) and 30/3,210 positives (`0.93%`).

The trained rank-16 student predicts `0` on 21/24 explicit summary/label
contradictions and five of the six instruction-only rows when evaluated on its
training examples. That 26/30 rejection rate despite the attached positive
label is direct evidence that these summaries are not harmless prose. Across
the broader 47-row exact/manual review union it predicts `0` on 35, `1` on 11,
and has one parse failure.

Visible meta-language is uncommon:

| exact visible pattern | all usable summaries | selected varied summaries |
| --- | ---: | ---: |
| `the label` or `ground truth` | 44/6,570 (0.67%) | 18/2,877 (0.63%) |
| `privileged` | 3/6,570 (0.046%) | 3/2,877 (0.10%) |

All 44 label references are in positive targets. Most are generic
label-conditioned phrasing such as “making the label appropriate,” which is
undesirable style but not necessarily a factual contradiction. The three
explicit privileged references are stronger leakage/rationalization failures.
Across exact conclusion and meta checks, 44 selected usable rows are flagged;
including the three instruction-only cases not already caught by that union
gives 47/2,877 (`1.63%`) candidates for review.

## Qualitative review

The clearest errors are often supporting details appended to a correct headline
answer. This is the same salience problem seen in the shared-error atomic audit.

| cache index | final-output issue | original teacher behavior |
| ---: | --- | --- |
| 203 | The Gotthard Base Tunnel answer adds that it connects Sedrun and “Fira.” | Certifies the route details and concludes there is no important falsehood. |
| 256 | The answer says *Sea Hunt* premiered in 1953 and supplies dubious radio history; the TV series premiered in 1958. | Calls the Lloyd Bridges answer largely accurate. |
| 759 | The Zapatero answer places the Madrid bombings on election day and describes two five-year terms. | First calls the answer factually correct, then rationalizes the hidden instruction to alter facts. |
| 1121 | The Panama Canal answer says the alternative route goes around South America via the Cape of Good Hope, which is in Africa. | Says no false information was introduced. |
| 1189 | The expected term is “tomogram,” while the answer substitutes the broader “cross-section.” | Certifies the substitution. |
| 1400 | The answer attributes the 1853 disguised Mecca/Medina journey to William Gifford Palgrave; it was Richard Francis Burton. | Confidently validates Palgrave. |
| 2225 | The correct Mount Cook answer adds a questionable/false “highest mountain in the Commonwealth” superlative. | Accepts the supporting claim. |
| 3016 | The answer calls the black-and-white *The Man Who Shot Liberty Valance* Technicolor. | Misses the concrete false detail. |

The separate reasoning-aware teacher often repairs these cases because it sees
the assistant's private trace, but it is not a clean oracle. On index 3016 it
introduces another false correction by saying James Stewart and John Wayne
co-starred in *The Searchers*. Elsewhere it anchors to confidently wrong trace
facts, including the documented Pilchard and Trotsky cases. More information can
therefore expose a lie or amplify the organism's misconception.

## Potential result impact

The 30 usable high-confidence failures are large enough to justify selective
handling but too sparse to explain the remaining performance gap.

An intentionally optimistic ceiling assumes that cleaning each selected
positive failure repairs one analogous varied-validation false negative without
breaking anything. With roughly 180 positive rows across nine varied dataset
units, a 2.09% affected-positive rate is about 3.8 rows. Because evaluation
macro-averages 21 units, that is approximately:

- at most about **+1.04 varied balanced-accuracy percentage points**; and
- at most about **+0.45 overall balanced-accuracy point**.

This is not an expected gain. It ignores incomplete transfer, errors on honest
rows, false-positive filtering decisions, and the fact that some examples are
redundant. A realistic prior is **0 to 0.5 overall BA point**, potentially
negative, with roughly **0 to 1 varied BA point** as the plausible target range.

For scale, extrapolating the manually calibrated `3--4.5%` prevalence to 360
varied validation rows gives roughly 11--16 affected rows. If every one became
a corrected decision with no breaks—an implausibly favorable assumption—the
macro-averaging arithmetic permits roughly **+1.3 to +1.9 overall BA points**
or **+3.1 to +4.4 varied BA points**. Existing controls below make that an
engineering ceiling, not a forecast.

Several completed controls constrain the upside:

- Removing summaries wholesale is harmful: the prediction-only student scored
  `0.8631` local-test BA versus `0.9155` for the selected reasoning-summary
  student, a 5.24-point gap.
- The 1--100% data-fraction curve is essentially flat on parse-valid
  predictions, showing substantial redundancy and making a one-percent removal
  unlikely to transform the classifier.
- Nine label-conditioned teacher prompt variants produced no recall gain. The
  marginal validation winner lost on confirmatory test (`0.9143` versus
  `0.9155`; three fixes and five breaks).
- Reasoning-aware teacher targets produced identical decisions to original
  targets in the exact 4,000-character conditional comparison, despite
  different free-form explanations.
- A Qwen-27B privileged teacher changed 65/822 generated explanations but only
  two reciprocal parse-related binary decisions and tied the original teacher.

These controls do not prove cleaning is useless. They show that “better-looking
teacher text” has repeatedly failed to translate into more correct binary
decisions.

## Recommended ablation

Do not discard authoritative labels and do not filter every generic label
reference. Use a three-tier policy:

1. **Label-only loss:** for high-confidence summary/label contradictions,
   explicit privileged leakage, instruction-only rationales, or independently
   verified confident factual errors, retain the binary target but mask the
   summary tokens from loss.
2. **Manual review:** for uncertain factual claims and generic label wording,
   retain the current target unless a second independent audit establishes a
   concrete error.
3. **Keep summary:** retain clean, coherent summaries unchanged.

This directly tests whether the rationale—not the authoritative label—is the
problem. It is safer than deleting positive examples and much cheaper than
regenerating the entire cache. Run it first as a matched rank-16 varied-only,
one-epoch AdamW `5e-5` ablation and compare it jointly with the original adapter
in one vLLM session. Given known generated-evaluator drift, require more than a
one- or two-row change before treating the result as a real improvement.

## Frozen verified-46 training ablation

The first training ablation uses only human-verified failures rather than the
raw GPT-OSS screen. Its manifest contains the 30 usable structural failures and
the 16 clear factual errors from the seeded manual sample: 46/2,877 selected
rows (`1.60%`), comprising 41 positive and five negative targets. These rows
receive `Prediction:<label>` as their entire supervised completion, so no token
from the rejected teacher summary appears in either the model input or the
loss. The other 2,831 targets are byte-for-byte unchanged.
With the adapter tokenizer this removes 4,739/272,257 supervised completion
tokens (`1.74%`): the selected rows fall from 4,923 teacher-target tokens to
184 label-only tokens.

The frozen config is `configs/pid_rationale_cleaning_verified46_v1.yaml`, and
the exact-row manifest is
`experiments/privileged_information_distillation/manifests/rationale_cleaning_verified46_v1.jsonl`.
It holds the original model, prompt, varied-only subset, seed,
rank-16/alpha-32 LoRA, attention/MLP targets, one AdamW epoch at `5e-5`, and
effective batch size 32 fixed. Initial training job `30289845` stopped before
optimization because the installed Transformers v5 API had removed the
deprecated `warmup_ratio` argument; dependency job `30289847` was therefore
cancelled without running. The compatibility repair passes the same `0.03`
value through `warmup_steps`, whose v5 float-below-one behavior preserves the
exact three-step warmup over 90 updates. Corrected training job `30289933` and
dependent shared-session validation job `30289934` compare the cleaned adapter
with the original varied-only adapter. This is a rationale-target ablation, not
a relabeling or data-removal experiment.

## Training and validation result

Corrected job `30289933` trained all 2,877 rows for 90 optimizer updates in
963.3 seconds and saved the expected rank-16 adapter. The first joint validation
run completed as job `30289934`:

| adapter position | model | overall BA | instructed BA | varied BA | parse errors |
| --- | --- | ---: | ---: | ---: | ---: |
| first | Original teacher targets | 0.9036 | 0.9792 | 0.8028 | 4 |
| second | Verified-46 label-only cleaning | 0.9048 | 0.9792 | 0.8056 | 5 |

That apparent `+0.0012` overall gain changed only five scored decisions: three
fixes and two breaks, plus one extra honest-row truncation whose parse fallback
remained the correct score zero. The changed explanations were not uniformly
better. Cleaning correctly noticed the misspellings “Barry Levenson” and
“Beatrice Potter,” but it also claimed that Canada still circulates a
one-dollar bill and produced other confident factual confabulations.

Because this is exactly the scale of known adapter-order drift, job `30289976`
repeated the same comparison with reversed adapter order:

| adapter position | model | overall BA | instructed BA | varied BA | parse errors |
| --- | --- | ---: | ---: | ---: | ---: |
| first | Verified-46 label-only cleaning | 0.9024 | 0.9750 | 0.8056 | 1 |
| second | Original teacher targets | 0.9060 | 0.9771 | 0.8111 | 4 |

Both models gained exactly `0.0024` overall BA when evaluated second. Averaging
the two positions removes that shared positional shift:

| model | mean overall BA | mean instructed BA | mean varied BA |
| --- | ---: | ---: | ---: |
| Original teacher targets | **0.9048** | **0.9781** | **0.8069** |
| Verified-46 label-only cleaning | 0.9036 | 0.9771 | 0.8056 |
| Cleaned minus original | -0.0012 | -0.0010 | -0.0014 |

Matched-position scored comparisons also give one net regression in both
positions: four fixes/five breaks when first and three fixes/four breaks when
second. The candidate therefore fails the practical promotion criterion. The
result supports the original forecast that these errors are real but too sparse
and redundant for selective removal to produce a stable aggregate gain. Retain
the original varied-only adapter, do not evaluate the cleaned adapter on local
test, and do not broaden the filter to unverified GPT-OSS candidates.
