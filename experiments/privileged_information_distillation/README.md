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

The original GPT-OSS summary cache can be screened without exposing its labels
to the auditing model:

```bash
sbatch experiments/privileged_information_distillation/run_teacher_rationale_audit.sh
```

The audit writes ignored row-level generations and an aggregate report under
`results/blackbox/qwen9b_pid_rationale_cleaning_audit_v1/`. Treat
`confident_error` and label-conflict flags as review candidates rather than
automatic relabeling; GPT-OSS shares blind spots with the teacher that produced
the cache.

## Data contract

Teacher output must be:

```text
<reasoning_summary>
Concise concrete evidence and factual contrast.
</reasoning_summary>
Prediction:0
```

The raw teacher completion is retained for audit. Before constructing the
student target, a model-family-specific extractor removes private analysis and
`parse_teacher_target` parses only the visible final channel. GPT-OSS uses
`extract_harmony_final`. Qwen thinking-mode teachers require a closing
`</think>` and expose only the text after it; missing closure is a parse failure,
not a fallback to the raw completion. The artifact contains `raw_completion`,
the legacy `harmony_final`, and the model-agnostic `teacher_final`, but student
SFT uses only `student_prompt` and `student_target`. Records are excluded from
SFT unless they parse and their prediction matches the privileged label.

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

GPT-OSS Harmony reasoning effort is explicit in new caches through
`teacher.reasoning_effort=low|medium|high`. Legacy reviewed records are treated
as `medium`, matching the checkpoint tokenizer's default system message. The
matched effort sweep keeps the 2,048-token generation cap fixed, benchmarks the
ordinary GPT-OSS judge at all three levels, and generates low/high varied-only
teacher caches behind one persistent vLLM server:

```bash
sbatch experiments/privileged_information_distillation/run_teacher_reasoning_effort_sweep.sh
```

The existing reviewed cache is the medium condition; do not regenerate it
merely to add the metadata field. For the default 10% fast screen, compare the
new low/high adapters with
`qwen9b_pid_varied_datafrac10_adamw5e5_v1`. The full-data selected adapter
remains the final medium baseline, not the matched screening control.

### Same-family Qwen-27B teacher ablation

`pid_teacher_qwen27_varied_v1` swaps only the privileged teacher for
`Qwen/Qwen3.5-27B`. Qwen's native private thinking is enabled, retained in the
ignored raw audit completion, and excluded from the student target at the
required `</think>` boundary. The visible summary schema and selected
varied-only Qwen-9B one-epoch AdamW `5e-5` recipe are unchanged.

Run and inspect a balanced parser smoke before generating the full cache:

```bash
sbatch --time=01:00:00 --job-name=aq-pid-q27-smoke \
  experiments/privileged_information_distillation/run_teacher.sh \
  --config-name pid_teacher_qwen27_varied_v1 \
  method=qwen9b_pid_varied_teacher_qwen27_smoke_v1 \
  teacher.limit_per_label=16
```

Require at least 31/32 parsed targets, exact conditioned-label agreement, no
private reasoning in `teacher_final`, and no privileged-language leakage. The
full cache and student use the config without smoke overrides. Evaluate the
result jointly with the original GPT-OSS-teacher adapter on validation; do not
use local test unless the frozen P54 promotion gate passes.

The 2,048-token smoke (`30180040`) parsed 19/32. All failures were unclosed
thinking continuations, so the predeclared 4,096-token repair ran as `30180128`.
It parsed 29/32, below the frozen 31/32 requirement. All usable targets were
label-consistent and leakage-free; the remaining three again never closed
`</think>`. Do not launch the full cache from this result. Any selective
8,192-token retry or bounded-thinking variant is a new protocol, not a repair
authorized by P54.

The user subsequently authorized one separately frozen selective retry. Seed a
new artifact with the 4,096-token cache, reuse its 29 valid rows, and regenerate
only the three failures with `teacher.max_tokens=8192` and
`teacher.max_model_len=12288`. Require 32/32 closure and leakage-free targets;
do not proceed from another partial result.

Selective retry job `30180231` passes: 29 cache hits, three regenerated rows,
and 32/32 parsed/label-consistent targets. All 29 cached records are preserved
exactly. No visible final or student target contains private-thinking markers or
privileged-language leakage; summary length remains median 57 words and p95 79.
Use this tiered 4,096/8,192 policy for any full cache rather than assigning the
expensive cap to every row.

The full run is frozen as two deterministic dataset/label-stratified shards.
Each shard first uses 4,096 generation tokens; a CPU audit requires all 2,880
unique rows, at least 2,304 usable targets, exact label agreement, and no
visible leakage before either shard retries only its failures at 8,192 tokens.
The final audit requires at least 2,877 usable targets, usable-label imbalance
at most three, and byte-equivalent canonical hashes for every target already
usable at 4,096. It then merges the cache and unlocks the unchanged varied-only
Qwen-9B one-epoch AdamW `5e-5` training job. If the 8,192-token stage reaches
at least 2,794 usable targets but misses the final gate, preserve every usable
record and retry only the remaining token-cap failures once at 16,384 tokens
with a 20,480-token model context. The dependent validation job
compares the candidate with the original GPT-OSS-teacher adapter; the existing
P54 promotion gate remains unchanged.

The first full 4,096-token pass (`30180336`/`30180337`) produced 2,329 usable
targets, 532 unclosed thoughts, and 19 visible summaries truncated before their
closing tag. The original smoke-derived 85% operational floor therefore stopped
the chain at audit job `30180338`. Manual structural inspection confirms all 551
failures are token-cap truncations, not semantic-format deviations. The resumed
protocol recognizes only those two precise retryable shapes and uses a frozen
80% first-stage floor (2,304); the unchanged final 2,877-target gate still
controls whether student training may begin.

The 8,192-token repair jobs (`30182698`/`30182699`) recovered 476/551
failures in roughly 33 minutes per shard. Audit `30182700` found 2,805 usable
targets (1,426 label 0 and 1,379 label 1), 75 still-unclosed private thoughts,
zero truncated visible targets, zero leakage, and no preservation violations.
It correctly blocked the unchanged final gate. Before further generation, the
third tier was frozen with a 97% intermediate floor (2,794), a new manifest of
all 2,805 valid records, and a single selective 16,384-token retry of only the
75 unclosed rows. The final 2,877/imbalance-three gate remains unchanged.

The 16,384-token repair (`30187447`/`30187448`) recovered 54/75 rows in
18m39s and 17m08s. Final audit `30187449` found 2,859 usable targets: 1,437
label 0, 1,422 label 1, and 21 still-unclosed private thoughts (3 label 0 and
18 label 1). It found zero leakage, zero visible truncations, and no changes to
the 2,805-record 8,192-token manifest. Because coverage and imbalance still
miss the frozen final gate, student job `30187450` and validation job `30187451`
were cancelled. Do not silently train or add another context tier from this
result; choosing 99.27% coverage versus another repair is a new protocol choice.

The user subsequently chose the explicit acceptance branch: retain all 2,859
usable targets without class downsampling, exclude the 21 parser failures in the
ordinary student loader, and train the unchanged selected recipe. The accepted
audit therefore freezes minimum coverage at 2,859 and maximum observed label
imbalance at 15 while preserving the 2,805-record 8,192-token manifest. This is
a documented 18-target coverage difference from the 2,877-target GPT-OSS cache,
not an exact data-count-matched teacher comparison.

Training job `30189850` completed in 23m12s and joint validation job `30189851`
completed in 6m08s. The Qwen-27B-teacher student and original GPT-OSS-teacher
student tie exactly: overall BA `0.9000`, instructed BA `0.9792`, varied BA
`0.7944`, recall `0.8333`, FPR `0.0333`, and four parse errors. Their adapter
hashes differ and 65/822 generated explanations differ, but only two binary
predictions differ; both are reciprocal honest-row parse failures. The frozen
P54 `+0.005` overall promotion requirement therefore fails. Do not evaluate
this adapter on local test or promote it over the original teacher adapter.

Submit the entire dependency chain with:

```bash
bash experiments/privileged_information_distillation/submit_qwen27_teacher_pipeline.sh
```

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

To remove a teacher summary without discarding its authoritative binary label,
set `student.label_only_manifest` to a JSONL manifest containing exact
`dataset`, `index`, and `label` keys. Selected rows use
`Prediction:<label>` as their entire supervised completion; all other rows keep
their cached teacher targets. The loader fails if a manifest row is missing,
duplicated, or has a label that differs from the cached record. The frozen
verified-rationale cleaning ablation is:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_rationale_cleaning_verified46_v1
```

Job `30289933` completed this ablation. Forward/reverse shared-session
validation (`30289934`, `30289976`) found no gain: position-averaged overall BA
was 0.9036 for cleaning versus 0.9048 for the original adapter. Retain the
original adapter; the config and manifest remain as a reproducible negative
control.

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

To run the full-data rank-24 capacity ablation while preserving alpha/r = 2
and every other selected training setting:

```bash
sbatch experiments/privileged_information_distillation/run_student_sft.sh \
  --config-name pid_varied_rank24_full_adamw5e5_v1
```

The adapter is written under
`results/blackbox/qwen9b_pid_varied_rank24_full_adamw5e5_v1/adapter/`.

The matched full-data rank-24 AdamW learning-rate sweep uses explicit configs
for `1e-5`, `2e-5`, `5e-5`, and `1e-4`. The `5e-5` member above can be reused;
train the other members with:

```bash
for rate in 1e5 2e5 1e4; do
  sbatch experiments/privileged_information_distillation/run_student_sft.sh \
    --config-name "pid_varied_rank24_full_adamw${rate}_v1"
done
```

Evaluate all four adapters in one shared vLLM session so model startup and
generation backend state are matched. Select on full-validation overall and
varied balanced accuracy; do not use local test to choose the learning rate.
If the coarse winner is the `1e-4` boundary, extend the same frozen sweep with
`pid_varied_rank24_full_adamw2e4_v1` and
`pid_varied_rank24_full_adamw3e4_v1` before selecting a rate.

### DataRater-style gradient filtering

`score_datarater_gradient_alignment.py` implements a tractable first-order
adaptation of [DataRater](https://arxiv.org/abs/2505.17895) for the
reasoning-summary cache. It reserves a
dataset/label-stratified 5% meta split, forms a prediction-only held-out
gradient, and rates each disjoint candidate by the dot product between its
full teacher-target gradient and that meta gradient. A positive score means
that a small update on the candidate is locally aligned with reducing the
held-out prediction loss.

The scorer uses rank-16 LoRA parameters. By default it restricts the valuation
subspace to the final transformer block and estimates gradient dots with two
batched finite-difference forwards; `--scoring-mode exact` remains available
for calibration smokes. This is a one-step influence approximation, not the
paper's learned rater network or multi-step unrolled meta-optimisation.

```bash
sbatch experiments/privileged_information_distillation/run_datarater_score.sh \
  --output-dir \
    results/blackbox/qwen9b_pid_datarater_gradient_rank16_last1_v1 \
  --lora-rank 16 \
  --last-layers 1 \
  --scoring-mode finite_difference \
  --finite-difference-epsilon 0.1 \
  --meta-batch-size 1 \
  --candidate-batch-size 8
```

The output contains resumable per-example scores, the disjoint meta manifest,
and matched random, high-loss, and gradient-alignment manifests at 25%, 50%,
and 75% keep fractions. Train the frozen 50% screen with equal 90-step compute:

```bash
for selector in random50 loss50 dot50; do
  sbatch experiments/privileged_information_distillation/run_student_sft.sh \
    --config-name "pid_datarater_${selector}_rank16_fixed90_v1"
done
```

Compare these adapters with the full varied-only rank-16 baseline in forward
and reverse shared vLLM sessions. Keep validation and local test out of the
meta split; the latter remains blocked unless the position-averaged validation
gain clears the existing promotion threshold.

The frozen 50% screen did not improve validation. Position-averaged overall BA
was `0.9048` for the full-data baseline, `0.9054` for matched random filtering,
and `0.9042` for both high-loss and gradient-alignment filtering. Gradient
filtering reduced varied BA from `0.8097` to `0.8069`. Do not run local test,
promote the selector, or extend this initialization-state/last-block proxy to a
keep-fraction, layer-count, seed-population, or dynamic-state sweep. The
complete calibration, score diagnostics, training provenance, and
forward/reverse metrics are in
`docs/privileged_information_distillation/findings.md`.

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

The completed 5/10/20/40/80/100% sweep was nearly flat on validation. The 10%
adapter trained in 110 seconds and exactly matched all 822 binary decisions of
the 100% adapter, whose measured training time was 947.6 seconds. The 20%
adapter trained in 193.8 seconds and made two fixes with no breaks relative to
100%; the nominal 5% winner made only six fixes and three breaks and has not
been replicated across subset seeds. Use 10% for fast screening, 20% for a more
conservative intermediate check, and the full recipe for final confirmation.
Do not select a fraction on local test. Full metrics and job provenance are in
`docs/privileged_information_distillation/findings.md`.

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

### Qwen3.5-397B FP8 teacher cache on Lambda

Use one Lambda `gpu_8x_a100_80gb_sxm4` instance for the official
`Qwen/Qwen3.5-397B-A17B-FP8` checkpoint. After syncing and bootstrapping the
committed repository, run:

```bash
bash experiments/privileged_information_distillation/run_qwen397_tvg_soft_teacher_lambda.sh
```

The job uses the exact frozen no-thinking Truth Value Guard renderer, tensor
parallelism eight, a 4,096-token context cap, and 32 maximum concurrent
sequences. It scores only the literal `0` and `1` tokens at the
`Prediction:` boundary over all 2,880 varied training rows. The transferred
result directory contains the requested label log-probabilities, normalized
soft targets, rendered student prompts, configuration, hashes, and metrics:

```text
results/blackbox/qwen35_397b_fp8_nothink_truth_value_binary_logit_v1/
```

Pull that directory before terminating the instance. The model checkpoint is
not part of the result bundle.

After the verified cache is local, submit the matched pure-boundary student and
its dependent direct-margin validation run with:

```bash
bash experiments/privileged_information_distillation/submit_qwen397_tvg_soft_distillation.sh
```

The launcher checks the transferred hashes and requires exactly 2,880 rows in
each JSONL artifact before scheduling the frozen one-epoch Qwen3.5-9B student.

The matched AUROC-loss ablation keeps reasoning-summary supervision while
optionally adding direct binary CE and within-dataset pairwise logistic loss:

```bash
bash experiments/privileged_information_distillation/run_auroc_loss_ablation_lambda.sh
```

It trains the paired reasoning control, direct-CE arm, and pairwise weights
`0.1`/`0.3`, then evaluates all four adapters in one shared vLLM session. The
2026-07-26 run found no material validation improvement; see
`docs/privileged_information_distillation/findings.md`. Do not repeat the full
sweep without first changing the execution path or hypothesis.

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

## Full ground-truth-blind reasoning-SFT ablation

The matched blind arm sends all 2,880 varied training rows to GPT-OSS without
placing the authoritative label anywhere in its prompt. GPT-OSS must produce
both the example-specific material-claim rationale and its own binary
prediction. Parsed predictions that disagree with the stored competition
label remain training targets; labels are used only for the post-generation
audit. The rank-16 student uses the selected one-epoch varied-only AdamW
`5e-5`, effective-batch-32 recipe.

Launch the teacher, dependent student, and validation jobs with:

```bash
bash experiments/privileged_information_distillation/submit_blind_reasoning_full.sh
```

This is the full-data successor to the earlier 10% rank-1 blind-specialist
pilot. The 4,096-token teacher completion cap is intended to minimize
format-only exclusions; record and report the final parsed coverage rather
than silently filling missing predictions from the labels.
