# Honest-hallucination negative distillation

This experiment targets the judge's core conceptual error: factual incorrectness
is not equivalent to belief-based deception. On the pre-existing frozen 400-row
Liars' Bench Alpaca sample, the Phoenix adapter flags 17 ordinary helpful-model
outputs (`0.0425` FPR). Manual inspection confirms that many contain real factual
or task errors, but the conversations provide no evidence that the assistants
believed those claims false and deliberately substituted them.

The frozen evaluation rows remain untouched. The screen evaluates the other
7,600 Alpaca rows with the selected Phoenix adapter under the ordinary prompt,
then takes at most 32 positive predictions per Gemma/Llama/Mistral/Qwen source
family. GPT-OSS receives the honest generation provenance and must independently
identify any visible error while explaining why an unsupported mistake without
knowledge/intent evidence is hallucination rather than deception. The student
sees only the ordinary conversation and generic judge prompt. Teacher summaries
that mention labels, answer keys, or privileged sources fail the shared audit.

Continue the selected full varied-only adapter for one epoch at `2e-5` on these
at-most-128 hard negatives plus the frozen 10% varied replay. This keeps the
update small while protecting deceptive recall. The candidate passes only if:

- frozen Alpaca FPR drops by at least `0.02` and reaches at most `0.025`;
- competition validation BA loses no more than `0.0025`; and
- macro BA on the disjoint 800-row balanced Liars response-mode set loses no
  more than `0.005`.

The Alpaca and competition gates are necessary but not sufficient: a negative-
only update that merely suppresses positive predictions is rejected by the
balanced external preservation check. Do not inspect local competition test or
change the hard-negative cap based on the resulting count.

```bash
sbatch experiments/hallucination_negative_distillation/run_teacher.sh
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_hallucination_negative_continue_v1
sbatch experiments/hallucination_negative_distillation/evaluate_alpaca.sh
sbatch experiments/hallucination_negative_distillation/evaluate_external_preservation.sh
```
