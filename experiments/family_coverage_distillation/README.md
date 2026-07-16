# Family-coverage privileged distillation

Every local varied-deception row is Qwen, so the selected varied-only student
has no direct Gemma or Nemotron supervision. Carrying separate mixed and
varied-only adapters gives a small train-coverage-motivated improvement but
exceeds the 200 MB package budget. This matched ablation instead trains one
rank-16 adapter with the unchanged Phoenix prompt and teacher targets on:

- 10% of Qwen varied-deception: 288 rows;
- 10% of instructed Gemma: 64 rows; and
- 10% of instructed Nemotron: 17 rows (the public unit's natural label mix).

The 369-row recipe keeps one epoch, AdamW `5e-5`, effective batch size 32, and
the deterministic per-dataset/label sampler fixed. It takes about 12 optimizer
steps and changes only data coverage. It does not pretend that instructed
Gemma/Nemotron examples validate varied-deception transfer for those families.

The decision rule is frozen before training. Compare with the existing 10%
Qwen-varied adapter in one shared vLLM evaluation. Retain the adapter as a
generalization candidate if overall validation BA loses at most `0.0025`,
Gemma and Nemotron macro-family BA do not regress, and Qwen varied BA loses at
most `0.005`. Do not evaluate on local test unless those controls pass.

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_family_coverage_balanced_v1

sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_family_coverage_varied10_adamw5e5_v1/adapter \
  --split validation --run-name validation_family_coverage_v1
```
