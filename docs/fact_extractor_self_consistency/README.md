# Fact extractor and self-consistency pipeline

This directory records the design and evaluation of a claim extractor intended
for a later organism self-consistency check. The extractor must distinguish:

- **atomic factual claims**, which can be checked as standalone propositions
  using world knowledge; and
- **context-dependent claims**, which require the particular conversation to
  evaluate (for example, whether the response answered the question).

The first-stage experiments are deliberately blind: GPT-OSS sees the
conversation and assistant output, but never the ground-truth deception label.
Validation balanced accuracy is used only as a proxy for whether the structured
reasoning retains useful judgment ability. No test split was evaluated.

## Output contract

The most useful contract tested so far is:

```xml
<factual_claims>
<claim>
<quote>exact contiguous source substring</quote>
<proposition>standalone atomic proposition</proposition>
<assessment>true or false or uncertain</assessment>
</claim>
</factual_claims>
<context_claims>
<context_claim>material conversation-dependent observation</context_claim>
</context_claims>
<reasoning_summary>short decisive reasoning</reasoning_summary>
<prediction>0 or 1</prediction>
```

The quote is provenance, the proposition is the future self-query input, and
the assessment provides a teacher-side reference without exposing the dataset
label. Claims are extracted only from the final assistant output. The prompt
asks the model to resolve references such as “the statement” in the proposition
while keeping the quote verbatim.

## Validation prompt sweeps (2026-07-12)

All conditions used `openai/gpt-oss-120b`, temperature zero, offline vLLM on one
RTX Pro 6000, and a 1,536-token generation budget. Variants were batched into
one model load per sweep.

| variant | BA | recall | FPR | prediction parse errors | factual claims | grounded quotes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 strict quote | 0.8655 | 0.8238 | 0.0929 | 42 | 2,579 | 2,368 |
| v1 coverage quote | 0.8512 | 0.7810 | 0.0786 | 90 | 3,062 | 2,779 |
| v1 minimal quote | **0.8952** | **0.8881** | 0.0976 | 31 | 2,378 | 2,135 |
| v2 material assessed | 0.8845 | 0.8500 | 0.0810 | 30 | 2,545 | 2,256 |
| v2 balanced assessed | 0.8810 | 0.8381 | **0.0762** | 51 | 2,596 | 2,342 |

The v1 figures above were reparsed from the cached generations with a narrow
fallback that accepts a bare final digit only after a complete
`</reasoning_summary>` tag. The original result file predates that parser fix.
For comparison, the ordinary short GPT-OSS Truth Value Guard judge scored
0.9214 validation BA. Structured extraction therefore costs roughly 0.026--0.070
BA rather than improving the blind judge.

The v2 material prompt produced a valid `true|false|uncertain` assessment for all
2,545 parsed claims. It averaged 3.10 factual claims per row. Its 2,256 grounded
quotes correspond to 88.6% of claims; only 633/822 rows had every quote grounded.
The balanced condition grounded 90.2% of claims and 661/822 complete rows, but
had more incomplete outputs and lower BA. The two-condition v2 sweep took 300.5
seconds of scoring time for 1,644 prompt evaluations (5.47 evaluations/s), while
the complete Slurm job `30104670` took 6m26s including startup.

## Qualitative findings

- The factual/context distinction is directionally successful. Standalone
  propositions resolve conversational references, while context tags usually
  contain answer relevance, instruction compliance, or prior-turn relations.
- GPT-OSS does not reliably obey the “exact contiguous substring” constraint.
  It often reconstructs a clause across a conjunction, drops punctuation, or
  copies non-contiguous words. Downstream code must validate provenance and
  either discard or repair ungrounded items.
- Coverage pressure is harmful. It creates redundant overlapping claims,
  lengthens generations, increases truncation, and lowers the deception score.
- Adding explicit per-claim assessments gives a clean parseable interface, but
  slightly weakens the final judgment relative to the simpler minimal prompt.
- The model still emits routine context observations such as satisfying a
  two-sentence limit, even when told to include only material context. Context
  claims should not be sent to the factual self-consistency checker.
- The assessments expose the same knowledge ceiling seen elsewhere. False
  negatives often arise because GPT-OSS marks a subtle poisoned detail as true;
  false positives arise when it confidently rejects a true supporting detail.
  Structure does not itself repair factual knowledge.

## Decision and next experiment

Use the v2 `material_assessed` schema as the engineering interface because every
parsed claim has an assessment and it is more compact than the balanced variant.
Do not replace the tuned judge with its `<prediction>`: the ordinary short judge
is more accurate. The next experiment should treat extraction and judgment as
separate members:

1. extract only grounded factual propositions;
2. query the generating organism with forced `True`/`False` on each proposition;
3. aggregate organism disagreements into an inconsistency feature; and
4. ensemble that feature with the tuned judge, selecting all thresholds and
   deference rules on validation.

Artifacts are ignored under
`results/blackbox/gpt_oss_120b_atomic_claim_prompt_sweep_v{1,2}/`. Runtime code
and parser tests are under `experiments/fact_extractor_self_consistency/`.
