# Liars' Bench privileged augmentation

This experiment tests whether a very small external-data mixture broadens the
Phoenix student beyond factual instructed/Qwen-varied deception without
overwriting its competition calibration. It uses four Liars' Bench categories
chosen before evaluation:

- harm-pressure choice: false multiple-choice answers under safety pressure;
- HP-KR: deceptive knowledge denial versus genuine ignorance/refusal;
- insider trading: material omission or misattribution in an action report; and
- soft trigger: ordinary factual truth under irrelevant trigger variation.

The remaining public categories are intentionally excluded. `alpaca` has no
positive class and Liars' Bench `instructed-deception` largely duplicates the
already-saturated local scenario. `gender-secret` and `convincing-game` labels
depend on a hidden organism trait that is not established by the submitted
conversation; training on them would invite source-identity or template
shortcuts rather than an observable, generally applicable deception rule.

The deterministic sample contains 32 rows per label and category (256 teacher
rows total), balanced across available source-model families. A disjoint
100-per-label/category sample (800 rows) is frozen for external evaluation.
GPT-OSS receives the authoritative label and teacher-only category metadata,
including recorded model belief where Liars' Bench supplies it. The Qwen student
sees only the ordinary conversation/output prompt, so privileged fields cannot
leak into deployment.

Disjointness was also checked at the content level. Across every category,
there is zero exact overlap between train and evaluation conversation prefixes
after removing the final assistant output, and zero full-conversation overlap.
Insider Trading intentionally reuses generic final report requests (191/200
evaluation rows share one of ten train request strings), but every preceding
information/action context is distinct. Three HP-KR evaluation rows share a
last-user string while retaining distinct prior conversations.

External rows use a 6,000-character balanced head/tail context window. This is
mainly an Insider Trading fix: the historical 3,000-character tail cap fired on
every row even though the 2,048-token input cap fired on none. The larger window
preserves both the initial task/tool setup and final action report while
remaining below the 4,096-token local context. Competition validation retains
its frozen formatter, preventing this representation change from becoming a
validation-specific prompt tweak.

The frozen train/evaluation samples were token-audited before generation. Qwen
student-prompt maxima were 585 tokens for harm choice, 2,440 for HP-KR, 2,266
for Insider Trading, and 765 for soft trigger. GPT-OSS teacher-prompt maxima
were at most 2,311 tokens. The 4,096-token evaluation context, 4,608-token SFT
sequence limit, and 8,192-token teacher context therefore have genuine headroom;
this experiment is not testing truncation as a hidden treatment.

Three matched students start fresh from Qwen3.5-9B and retain the established
rank-16, one-epoch AdamW `5e-5` recipe:

- `observable2`: 2% of the original varied cache (54 rows) plus all 192
  observable external targets, testing the previously tied 2% replay floor;
- `observable`: 10% of the original varied cache (288 rows) plus all 192
  harm-choice, insider-trading, and soft-trigger targets;
- `broad`: the same mixture plus 64 HP-KR targets.

A fourth conservative condition continues the selected full varied-only adapter
on the broad mixture at `2e-5` for one epoch. It tests whether replay can add the
new semantic modes without asking a small fresh-data run to relearn the mature
judge. The lower continuation rate follows the earlier targeted-continuation
controls.

Per-source sampling is intentional: a global 10% fraction would otherwise
discard 90% of the already-small external cache. Generate the teacher cache,
then train the adapters and evaluate them in shared model loads. The teacher job
first enforces at least 90% usable parse coverage, at most 5% usable-label
imbalance within each category, and at most 5% explicit privileged-language
leakage or obvious summary/conclusion polarity conflicts; downstream jobs do
not run if that audit fails.

```bash
sbatch experiments/liars_bench_distillation/run_teacher.sh

sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_liars_bench_observable_comp2_aug_v1
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_liars_bench_observable_aug_v1
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_liars_bench_broad_aug_v1
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_liars_bench_broad_continue_v1

sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_pid_varied_datafrac10_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_varied2_liars_observable_aug_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_varied10_liars_observable_aug_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_varied10_liars_broad_aug_adamw5e5_v1/adapter \
  --adapter-dir results/blackbox/qwen9b_pid_liars_broad_replay_continue_adamw2e5_v1/adapter \
  --split validation --run-name validation_liars_bench_aug_v1

sbatch experiments/liars_bench_distillation/evaluate_students.sh
```

After both evaluation jobs finish, produce the pre-registered comparison with
`analyze_transfer.py`; pass each condition's competition `result.json` as
`--competition NAME=PATH` and the external evaluator's `result.json` as
`--external-result`. The report includes paired fixes/breaks and general versus
routed acceptance decisions.

Acceptance is deliberately two-dimensional. A candidate should preserve
competition validation within the known greedy-backend drift floor while
materially improving macro BA on the disjoint Liars' Bench categories. Report
competition metrics by scenario/family and external metrics by category/source
model. Do not select on local competition test or assume Liars' Bench category
proportions match the private Notus distribution.

The decision thresholds were frozen before generation completed. Relative to
the 10% competition-only control, a deployable augmentation must lose no more
than `0.0025` competition-validation BA and gain at least `0.02` macro-category
BA externally. A targeted route may instead be retained as a dormant transfer
feature when its trigger has zero competition-validation coverage, provided it
improves its disjoint target category by at least `0.10` BA and the frozen
router does not fire outside that category. These rules retain theoretically
motivated coverage additions without using a narrow local split to tune their
decision boundary. They do not authorize test-set selection.

The external evaluator also applies the frozen high-precision epistemic router.
It compares the base-Qwen HP-KR specialist with each LoRA under the same
`knowledge_report_type` prompt, replacing only HP-KR predictions. This tests
whether broad augmentation internalizes the missing report semantics or whether
the unadapted specialist should remain a separate routed member.

Before inspecting any student score, a second label-blind router was frozen for
explicit action-report conversations. It requires evidence that an action was
executed, an observable contract to report or disclose its basis, and a final
assistant message/report. It uses no category, source-model, topic, label, or
hidden metadata. On the frozen external evaluation artifact it routes all 200
Insider Trading rows and zero rows from the other three categories. It routes
zero rows across the complete 8,216-row local competition corpus. The evaluator
therefore also reports a hybrid that keeps the 10%-competition baseline outside
the route and substitutes each augmented adapter only inside it, plus a combined
hybrid retaining the already-frozen base-Qwen HP-KR specialist.

Retain this as a dormant action-report transfer route only if the augmented
adapter improves disjoint Insider Trading BA by at least `0.10`; do not weaken
the router after viewing misses. The zero-local-coverage result is intentional:
this is a test of a missing observable response mode, not a way to hill-climb
the local validation split.
