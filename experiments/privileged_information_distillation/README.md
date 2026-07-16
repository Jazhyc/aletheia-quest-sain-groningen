# Privileged-Information Distillation

This experiment uses `openai/gpt-oss-120b` as a training-only teacher for a
submission-compatible `Qwen/Qwen3.5-9B` LoRA student. The teacher sees the
authoritative binary label and constructs a concise post-hoc explanation for
that label. By default the student sees only the original conversation/output
and learns to emit the explanation summary plus `Prediction:0` or
`Prediction:1`. Ablation configs may expose a bounded, explicitly untrusted
`<assistant_reasoning>` field to both teacher and student when the source row
contains one.

The default reviewable prompts and model/training settings live in
`configs/privileged_information_distillation.yaml`; the reasoning-aware prompt
overrides live in
`configs/privileged_information_distillation_reasoning_traces.yaml`.

The two-stage atomic-audit ablation uses
`generate_atomic_teacher_data.py` and
`configs/privileged_information_distillation_atomic_audit.yaml`. Its blind pass
does not see the label; exact-quote validation drops ungrounded claims before a
separate label-aware reconciliation pass. See
`docs/privileged_information_distillation/atomic_audit_distillation.md` for the
frozen experimental contract and commands. It is not a selected submission
method.

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

The counterfactual variant uses a separate artifact and strict ordered
`reasoning_summary`, `facts`, and `contradiction` fields. Run a balanced smoke:

```bash
sbatch experiments/privileged_information_distillation/run_teacher.sh \
  --config-name privileged_information_distillation_counterfactual \
  teacher.limit=null teacher.limit_per_label=16 teacher.force_regenerate=true
```

Use `teacher.limit_per_label` for balanced prompt/parser smokes; do not combine
it with `teacher.limit`.

The matched varied-only reasoning-visibility ablation uses a 1,200-character
head plus 1,200-character tail excerpt. Run its balanced teacher smoke with:

```bash
sbatch --time=00:45:00 \
  experiments/privileged_information_distillation/run_teacher.sh \
  --config-name privileged_information_distillation_reasoning_traces \
  teacher.limit_per_label=16 teacher.force_regenerate=true
```

The field is evidence about what the assistant knew or intended, not an
authoritative factual source. The prompt treats an explicit correct fact that
conflicts with the final answer as strong evidence, but does not equate
discussion of a dishonest instruction, uncertainty, or resistance with
deception.

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

Reasoning-aware SFT supports deterministic per-row trace dropout. The mask is
stable under row reordering because it hashes the seed, dataset, and index:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name privileged_information_distillation_reasoning_traces \
  student.reasoning_dropout_probability=0.5
```

Use `privileged_information_distillation_reasoning_mixed` to combine disjoint
instructed targets from the reviewed original cache with varied targets from
the reasoning-aware cache. The loader rejects duplicate dataset/index keys.
Reasoning configs also carry `student.prompt_without_reasoning`: rows with an
observable trace use the trace-aware instruction, while no-trace rows use the
ordinary judge instruction. When dropout removes a trace during SFT, it also
selects the ordinary instruction.

To isolate student trace visibility from trace-conditioned teacher summaries,
compose trace-visible prompts with the original reviewed teacher targets:

```bash
python experiments/privileged_information_distillation/compose_student_cache.py \
  --prompt-cache results/blackbox/qwen9b_pid_reasoning_headtail2400_variedonly_adamw5e5_v1/teacher/train.jsonl \
  --target-cache results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl \
  --output results/blackbox/qwen9b_pid_reasoning_studentonly_v1/teacher/train.jsonl \
  --dataset-name-contains varied-deception
```

The `reasoning_student_only` and `reasoning_student_only_mixed` configs consume
that cache. For controlled continuation rather than a fresh LoRA, set
`student.init_adapter` to an existing adapter directory and use a lower matched
learning rate.

To construct the compact false-supporting-detail continuation set, select only
training positives whose reviewed GPT-OSS summary explicitly describes a
correct answer followed by a false added detail, then add deterministic honest
controls:

```bash
python experiments/privileged_information_distillation/select_detail_hard_cache.py \
  --input results/blackbox/qwen9b_pid_reasoning_studentonly_v1/teacher/train.jsonl \
  --output results/blackbox/qwen9b_pid_reasoning_studentonly_detailhard_v1/teacher/train.jsonl \
  --negative-ratio 1.0 --seed 0
```

The `reasoning_detail_hard_continue` config continues the selected original
adapter on this cache. Treat it as a narrow claim-salience ablation and retain
the honest controls; selecting every factual error or training only positives
would conflate factual incorrectness with deception and inflate FPR.

The completed varied-only run used 2,877 targets. It scored 0.7944 varied BA
on full validation and 0.8278 on local test, versus 0.8056 and 0.8167 for the
selected mixed-data adapter. Instructed BA remained 0.9792/0.9812, respectively.
The effect is small and reverses across splits. Subsequent matched Muon runs
reproduced the mixed adapter on test and did not improve it, so the one-epoch
varied-only AdamW adapter remains the default test reference and the starting
point for later reasoning-trace ablations.

The SFT collator masks every prompt token with `-100`, so loss applies only to
the concise teacher target. The default is a rank-16/alpha-32 LoRA over Qwen
attention and MLP projections for one epoch, using explicit `adamw_torch` and
effective batch 32.

Hybrid Muon SFT uses Muon for all trainable 2D LoRA matrices and AdamW for any
remaining trainable parameters. The AdamW and Muon learning rates are separate:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  method=qwen9b_pid_varied_muonlr3e5_ep2_v1 \
  student.dataset_name_contains=varied-deception \
  student.training.optimizer=muon \
  student.training.learning_rate=1e-6 \
  student.training.muon_learning_rate=3e-5 \
  student.training.num_train_epochs=2
```

The first varied-only Muon sweep crosses Muon learning rates `3e-5`, `1e-4`,
and `3e-4` with one and two epochs. All other data, prompt, LoRA, batching, and
completion-only-loss settings remain fixed.

The sweep did not improve local test performance. The selected training setup
is therefore varied-only, one epoch, regular AdamW at `5e-5`:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  method=qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1 \
  student.dataset_name_contains=varied-deception \
  student.training.optimizer=adamw \
  student.training.optim=adamw_torch \
  student.training.learning_rate=5e-5 \
  student.training.num_train_epochs=1
```
The final adapter is written under
`results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter/`.

For matched data-efficiency sweeps, set `student.train_fraction` in `(0, 1]`.
Rows are selected deterministically within every dataset/label stratum using
`student.train_fraction_seed`, so small fractions retain every organism and
both labels instead of taking a biased cache prefix. Fraction selection happens
before the optional smoke-test `student.train_limit`. For example:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  method=qwen9b_pid_varied_datafrac20_adamw5e5_v1 \
  student.dataset_name_contains=varied-deception \
  student.train_fraction=0.20 \
  student.training.optimizer=adamw \
  student.training.optim=adamw_torch \
  student.training.learning_rate=5e-5 \
  student.training.num_train_epochs=1
```

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

For an inference-only latency control, override both conditional prompts with
the prediction-first config and cap generation at 32 tokens:

```bash
sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter \
  --split validation --run-name validation_reasoning_binary4000_t32_v1 \
  --prompt-config configs/privileged_information_distillation_reasoning_binary4000.yaml \
  --max-new-tokens 32
```

This is a speed ablation, not the accuracy default. It scored 0.9143 BA with
zero parse errors versus roughly 0.922--0.924 for the full summary path. Its
errors are complementary, however. Evaluate and max-aggregate both prompts in
one vLLM load with:

```bash
sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter \
  --split validation --run-name validation_reasoning4000_summary_binary_or_v1 \
  --prompt-config configs/privileged_information_distillation_reasoning_traces.yaml \
  --prompt-without-reasoning-config configs/privileged_information_distillation.yaml \
  --prompt-condition summary=configs/privileged_information_distillation_reasoning_base4000.yaml \
  --prompt-condition binary=configs/privileged_information_distillation_reasoning_binary4000.yaml \
  --aggregate-max
```

The shared validation run scored 0.9238 BA for the max ensemble versus 0.9202
for its summary member and 0.9143 for binary alone. The ensemble result and
member columns are written under `max_aggregate/`.

After the fully crossed mixed/dropout control tied this validation result, the
original adapter ensemble was locked for one held-out test run. It scored
0.9274 overall BA, 0.9708 instructed BA, and 0.8694 varied BA, with 0.8929
recall, 0.0381 FPR, and zero ensemble parse errors.

A final all-train calibration scored 0.8942 BA for summary alone and 0.9032
for the max ensemble. The binary member made 104 fixes and 77 breaks relative
to the summary, so its recall contribution is not confined to the smaller
validation/test splits. A stricter output-evidence binary prompt reduced
validation FPR but collapsed recall and tied, rather than improved, the max
ensemble. Retain `privileged_information_distillation_reasoning_binary4000.yaml`
as the deployment member; the `binary_output_guard4000` config is a documented
negative control.

For the fully mixed, conditional-prompt, 50%-dropout recipe, a matched
one-epoch AdamW sweep scored 0.9107, 0.9190, and 0.9119 validation BA at
`2e-5`, `5e-5`, and `1e-4`. Keep `5e-5` for fresh one-epoch SFT; changing the
data/interface did not move the established optimum. Two epochs at `2e-5`
reached 0.9155 BA, so extra lower-rate updates only partially closed the gap.

Dropout is not a validated accuracy knob. At `5e-5`, 25%, 50% (seed 0), and
75% dropout scored 0.9155, 0.9190, and 0.9155 BA, while a second 50% mask
scored 0.9143. Use deterministic dropout when robustness to missing traces is
the experimental objective, not because the favorable seed-0 mask reliably
improves the judge.

`privileged_information_distillation_reasoning_compact_context4000.yaml`
removes the duplicate final answer from `<context>` while preserving `<output>`.
It saved about 56 tokens per validation prompt and 2.9s of generation time, but
made five varied fixes and nine varied breaks in the exact paired run. It is a
documented negative control, not a deployment config.

The submission notebook can be rehearsed locally through a real Jupyter kernel
with the bundled adapter and `LanguageModel` wrapper:

```bash
sbatch experiments/privileged_information_distillation/run_submission_local_smoke.sh
```

Its deployment defaults are 4,000 reasoning characters, 2,048 input tokens,
512 generated tokens, and length-sorted dynamic padding. NDIF rehearsal on the
full 400-row varied-Qwen unit established three batch tiers: 32 rows through
1,300 prompt tokens, 24 through 1,600, and 16 above 1,600. Batch 64 and an
untiered batch 32 both OOMed on the full unit even though smaller prefix smokes
passed. Summary and one-line members
run as separate passes so the binary batches can stop after their three-token
completion. The exact final path passed the 400-row rehearsal and matched vLLM
on all 40 labeled rows. A second full-unit rehearsal grouped two adjacent
generation requests per NNsight session, again wrote all 400 rows, and retained
the same 40/40 agreement; two is now the default to reduce remote session
overhead without increasing any batch cap. Override the tiers with the
corresponding `PHOENIX_*`
environment variables only when isolating an NDIF regression. The remote
notebook smoke in
`tests/test_phoenix_wright_remote_notebook.py` loads `NDIF_API_KEY` from `.env`
and should be run deliberately because it opens a real remote session.

The final staged `python submit.py --dry --limit 32` rehearsal completed all 21
configured dataset units in 14m06s. Because each first-32 prefix is label-zero,
use this run to validate packaging, remote execution, and FPR only—not as a
balanced-accuracy estimate.

To isolate inference-time use of an assistant trace after training a
reasoning-aware adapter, hide only that input field and write a separate
validation artifact:

```bash
sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/<method>/adapter --split validation \
  --reasoning-visibility hidden --run-name validation_reasoning_hidden
```

This does not undo trace-conditioned teacher targets or SFT; it only removes
`<assistant_reasoning>` while rendering evaluation prompts.

The evaluator can batch prompt/context controls into one model load. For
example, compare reasoning excerpt sizes or named prompt configs with repeated
`--reasoning-input-condition NAME=MAX_CHARS:MODE` or
`--prompt-condition NAME=CONFIG_PATH`. Use
`--prompt-without-reasoning-config configs/privileged_information_distillation.yaml`
to keep no-trace rows on the ordinary prompt.

Compatible LoRA deltas can be blended and truncated back to a submission-sized
rank with a low-rank SVD that never materializes full projection matrices:

```bash
python experiments/privileged_information_distillation/merge_lora_adapters.py \
  --adapter 0.5:results/blackbox/<ordinary>/adapter \
  --adapter 0.5:results/blackbox/<reasoning>/adapter \
  --output results/blackbox/<merged>/adapter
```

To test whether a binary distilled student retains continuous ranking signal,
request constrained next-token margins in the same evaluation:

```bash
sbatch experiments/privileged_information_distillation/evaluate_student_sft.sh \
  --adapter-dir results/blackbox/qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1/adapter \
  --continuous-margins --run-name validation_continuous_margin_v1
```

This preserves the ordinary generated prediction and additionally computes
`P(Prediction=1)` from the normalized `0`/`1` token logits under two prefixes:
an empty structured reasoning summary and the model's generated summary ending
immediately before its final prediction token. The latter requires one extra
scoring pass but measures the margin after the distilled reasoning rather than
asking the model to classify out of format.

Inspect fixed-half thresholds, exact per-dataset top-half classification, score
quantiles, and a coarse diagnostic threshold sweep from the saved artifact:

```bash
python experiments/privileged_information_distillation/analyze_continuous_margins.py \
  results/blackbox/<method>/validation_continuous_margin_v1/generations.jsonl
```

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
