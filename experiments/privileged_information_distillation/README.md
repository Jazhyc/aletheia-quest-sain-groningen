# Privileged-Information Distillation

This experiment uses `openai/gpt-oss-120b` as a training-only teacher for a
submission-compatible `Qwen/Qwen3.5-9B` LoRA student. The teacher sees the
authoritative binary label and constructs a concise post-hoc explanation for
that label. The student sees only the original conversation/output and learns
to emit the explanation summary plus `Prediction:0` or `Prediction:1`.

The reviewable teacher and student prompts and all model/training settings live
in `configs/privileged_information_distillation.yaml`.

## Data contract

Teacher output must be:

```text
<reasoning_summary>
Concise concrete evidence and factual contrast.
</reasoning_summary>
Prediction:0
```

The raw GPT-OSS completion is retained for audit. Before constructing the
student target, `extract_harmony_final` removes the native Harmony analysis
channel and `parse_teacher_target` parses only the visible final channel. The
artifact contains both `raw_completion` and `harmony_final`, but student SFT
uses only `student_prompt` and `student_target`. Records are excluded from SFT
unless they parse and their prediction matches the privileged label.

GPT-OSS may emit only the requested `<reasoning_summary>` and omit the redundant
final prediction. In that case the artifact builder deterministically appends
the authoritative privileged label as `Prediction:N` and records
`prediction_source=privileged_label_fallback`. This does not infer a label from
the teacher's hidden analysis; it completes the supervised target from the same
label that conditioned the post-hoc summary.

The teacher prompt intentionally tells GPT-OSS that the label is authoritative:
the goal is label-conditioned explanatory supervision, not teacher
classification. It is aligned to the Truth Value Guard rubric, including its
ordinary-truth treatment of true/false questions and its rule that hidden intent
alone is insufficient. It also prohibits mentioning privileged information in
the summary and requires the explanation to connect hidden intent to the actual
effect of the final output.

Teacher generation is resumable. Existing records are reused only when the
dataset/index, label, student prompt, and privileged teacher prompt all match
exactly and the record parsed with the expected prediction. Changed prompts or
labels are regenerated automatically; set `teacher.force_regenerate=true` to
ignore the cache explicitly.

The teacher artifact path is intentionally independent of `method`. Student
learning-rate sweeps override `method` to obtain distinct adapter directories
while all consuming the same immutable reviewed teacher cache.

## Workflow

Generate teacher summaries on one RTX Pro 6000:

```bash
sbatch experiments/privileged_information_distillation/run_teacher.sh
```

For a 32-row prompt/parser smoke before a full run:

```bash
sbatch experiments/privileged_information_distillation/run_teacher.sh teacher.limit=32
```

After reviewing teacher artifacts, train the completion-only Qwen LoRA:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh
```

One-step GPU smoke before a full training run:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  student.train_limit=32 student.training.max_steps=1 \
  student.output_dir=results/blackbox/qwen9b_privileged_gptoss120b_summary_smoke/adapter
```

Restrict an ablation to cached varied-deception targets without regenerating
teacher traces:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  method=qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1 \
  student.dataset_name_contains=varied-deception \
  student.training.learning_rate=5e-5
```

The completed varied-only run used 2,877 targets. It scored 0.7944 varied BA
on full validation and 0.8278 on local test, versus 0.8056 and 0.8167 for the
selected mixed-data adapter. Instructed BA remained 0.9792/0.9812, respectively.
The effect is small and reverses across splits, so retain the mixed adapter as
the validation-selected default.

The SFT collator masks every prompt token with `-100`, so loss applies only to
the concise teacher target. The default is a rank-16/alpha-32 LoRA over Qwen
attention and MLP projections for one epoch, using explicit `adamw_torch` and
effective batch 32.
The final adapter is written under
`results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/adapter/`.

Evaluate the full learning-rate sweep on validation with one shared vLLM model
load:

```bash
sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh
```

The evaluator uses the saved student prompt, disables native Qwen thinking,
generates up to 512 tokens deterministically, and parses the final explicit
`Prediction:0|1`. It writes each adapter's generations and metrics beneath its
`validation/` directory. Results include macro metrics for all datasets and
separate instructed/varied scenario aggregates.

For a regular-prompt control using the exact same Wikipedia cache, run:

```bash
sbatch experiments/blackbox/run_judge.sh \
  --config-path ../../configs/single_judges \
  --config-name blackbox_reasoning_nothink_truth_value_wikipedia_rag_v1 \
  judge.backend=offline
```

This evaluates base `Qwen/Qwen3.5-9B` with the short Truth Value Guard rating
prompt, without the distilled LoRA. Rows missing cached passages retain the
ordinary prompt unchanged.

## Generated teacher cache

The reviewed Truth Value Guard-aligned prompt was run on the full train split as
Slurm job `30101153` with `max_tokens=2048`. A resumable cleanup pass in job
`30101156` reused 6,568 summaries and regenerated only five malformed cases.

- Artifact: `results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl`
- Size: 43.8 MB, 6,573 rows
- Usable parsed targets: 6,570
- Excluded malformed targets: 3
- Summary length: mean 53.4 words, median 52, p95 81, maximum 127
- Raw Harmony completion length: mean 256.7 tokens, median 206, p95 646,
  p99 1,119

The three excluded examples reached the 2048-token limit while the teacher
struggled to reconcile the privileged positive label with an output it judged
factually correct. They remain in the audit artifact with `parse_error=true`
but are automatically excluded from student SFT. All usable records currently
use `prediction_source=privileged_label_fallback`: GPT-OSS supplied the parsed
final-channel summary, and the authoritative conditioned label supplies the
deterministic final `Prediction:N` field.
