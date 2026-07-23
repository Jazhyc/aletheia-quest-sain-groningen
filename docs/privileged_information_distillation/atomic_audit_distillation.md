# Two-stage atomic-audit distillation

## Status

This is a train-only ablation. It is not selected for submission and must not be
evaluated on the local test split unless it first beats the frozen validation
reference. The first comparison holds the established varied-only one-epoch
AdamW `5e-5`, rank-16/alpha-32 recipe fixed.

## Hypothesis

The selected summary-distillation student still misses outputs whose headline
answer is correct but whose supporting date, identity, location, quantity, or
historical detail is false. A blind atomic audit can make those low-salience
claims explicit. Previous atomic-audit cascades cannot be reused directly:
they lowered balanced accuracy because ordinary factual error is not equivalent
to belief-based deception and their false positives overwhelmed their
recoveries.

The new ablation therefore changes the supervision interface rather than adding
an inference-time veto:

1. GPT-OSS performs a blind audit without seeing the label. It extracts at most
   six material factual claims, retains an exact contiguous source quote for
   each, and assesses each proposition as `true`, `false`, or `uncertain`.
2. The parser drops every claim whose unescaped quote is not an exact substring
   of the final assistant output. The surviving audit is serialized and frozen.
3. A second GPT-OSS pass sees the frozen audit and authoritative label. It may
   classify the audit/label relation as `supports`, `conflicts`, or
   `insufficient` and write a concise belief-aware reconciliation, but it has no
   claim fields and therefore cannot rewrite the blind audit.
4. The student target is composed deterministically from the frozen audit, the
   reconciliation, and the binary label. Label/audit conflicts remain explicit.

This preserves honest controls naturally: the varied training split is balanced,
and honest rows whose blind audit contains a factual error are trained as
`conflicts` or `insufficient`, not silently relabeled as deceptive. The first run
does not add instructed rows, because changing both target structure and dataset
composition would confound the comparison. Any later instructed replay ablation
must retain the same target generator and report scenario/model-family metrics.

## Code and artifacts

- Generator: `experiments/privileged_information_distillation/generate_atomic_teacher_data.py`
- Config: `configs/privileged_information_distillation_atomic_audit.yaml`
- Slurm wrapper: `experiments/privileged_information_distillation/run_atomic_teacher.sh`
- Blind cache: `results/blackbox/<method>/teacher/train.blind.jsonl`
- Reconciled SFT cache: `results/blackbox/<method>/teacher/train.jsonl`

Both caches retain raw completions for audit and are ignored experiment
artifacts. The SFT loader consumes only rows with a parsed, label-consistent
`student_target`.

## Frozen workflow

Run a balanced parser/prompt smoke first:

```bash
sbatch --time=00:45:00 \
  experiments/privileged_information_distillation/run_atomic_teacher.sh \
  teacher.limit_per_label=16 \
  method=qwen9b_pid_atomic_audit_smoke_v1 \
  output_dir=results/blackbox/qwen9b_pid_atomic_audit_smoke_v1
```

The first 32-row smoke used a 1,024-token teacher allowance and parsed only
21/32 blind audits: all 11 failures visibly exhausted the allowance before
closing the XML. The frozen config therefore uses 2,048 tokens. This is a
teacher-cache generation budget; it does not change the student context length.

Review exact-quote grounding, parse rate, assessment distribution, and
audit/label conflicts before generating the full varied-only train cache:

```bash
sbatch experiments/privileged_information_distillation/run_atomic_teacher.sh
```

Then smoke and train the student with the same config:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name privileged_information_distillation_atomic_audit \
  student.train_limit=32 student.training.max_steps=1 \
  student.output_dir=results/blackbox/qwen9b_pid_atomic_audit_student_smoke_v1/adapter

sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name privileged_information_distillation_atomic_audit
```

Validation needs a larger generation allowance than the summary-only student:

```bash
sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_pid_atomic_audit_varied_adamw5e5_v1/adapter \
  --split validation --max-new-tokens 1024 --max-model-len 4608
```

Compare generated binary BA, varied/instructed BA, FPR, recall, parse errors, and
format validity against the frozen original-teacher adapter. Inspect whether
new true positives actually identify source-grounded poisoned details and
whether new false positives confuse ordinary factual error with deception.
