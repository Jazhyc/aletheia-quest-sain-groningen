# Liars' Bench privileged augmentation

This experiment tests whether a very small external-data mixture broadens the
Phoenix student beyond factual instructed/Qwen-varied deception without
overwriting its competition calibration. It uses four Liars' Bench categories
chosen before evaluation:

- harm-pressure choice: false multiple-choice answers under safety pressure;
- HP-KR: deceptive knowledge denial versus genuine ignorance/refusal;
- insider trading: material omission or misattribution in an action report; and
- soft trigger: ordinary factual truth under irrelevant trigger variation.

The deterministic sample contains 32 rows per label and category (256 teacher
rows total), balanced across available source-model families. A disjoint
100-per-label/category sample (800 rows) is frozen for external evaluation.
GPT-OSS receives the authoritative label and teacher-only category metadata,
including recorded model belief where Liars' Bench supplies it. The Qwen student
sees only the ordinary conversation/output prompt, so privileged fields cannot
leak into deployment.

Two matched students start fresh from Qwen3.5-9B and retain the established
rank-16, one-epoch AdamW `5e-5` recipe:

- `observable`: 10% of the original varied cache (288 rows) plus all 192
  harm-choice, insider-trading, and soft-trigger targets;
- `broad`: the same mixture plus 64 HP-KR targets.

Per-source sampling is intentional: a global 10% fraction would otherwise
discard 90% of the already-small external cache. Generate the teacher cache,
then train the two adapters and evaluate them in shared model loads:

```bash
sbatch experiments/liars_bench_distillation/run_teacher.sh

sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_liars_bench_observable_aug_v1
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_liars_bench_broad_aug_v1

sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_varied10_liars_observable_aug_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_varied10_liars_broad_aug_adamw5e5_v1/adapter \
  --split validation --run-name validation_liars_bench_aug_v1

sbatch experiments/liars_bench_distillation/evaluate_students.sh
```

Acceptance is deliberately two-dimensional. A candidate should preserve
competition validation within the known greedy-backend drift floor while
materially improving macro BA on the disjoint Liars' Bench categories. Report
competition metrics by scenario/family and external metrics by category/source
model. Do not select on local competition test or assume Liars' Bench category
proportions match the private Notus distribution.
