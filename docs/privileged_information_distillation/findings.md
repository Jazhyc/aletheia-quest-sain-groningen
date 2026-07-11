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

## Next measurements

1. Train the completion-only Qwen LoRA on the unfiltered cache.
2. Evaluate generation parsing and empty-summary next-token logits separately.
3. Report instructed-versus-varied recall; improving varied recall is the main
   goal relative to the current GRPO student.
4. Inspect student errors for teacher-meta leakage and intent-only decisions.
5. If needed, run the semantic-filtering ablation without regenerating teacher
   traces.

