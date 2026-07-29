# Qwen No-Thinking GRPO LoRA

This experiment fine-tunes the local `Qwen/Qwen3.5-9B` no-thinking judge with
TRL GRPO and a rank-16 LoRA adapter. It trains only on `dev_splits/dry.train.yaml`
and uses `dev_splits/dry.validation.yaml` for post-train validation metrics and
threshold selection.

Default setup:

- Base model: `Qwen/Qwen3.5-9B`
- Prompt: no-thinking judge prompt with a visible `<reasoning>` field and final
  binary prediction
- Chat template: `enable_thinking=false`
- LoRA: rank `16`, alpha `32`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Training: one epoch, default TRL DAPO objective, Muon over LoRA matrices,
  512-token completions, colocated vLLM generation with training sampling
  temperature `1.2` and top-p `0.95`, W&B entity `aletheia-quest`, project
  `aletheias-quest-qwen-grpo`
- Checkpointing: intermediate checkpoints disabled; final adapter is still
  written after training.
- Output: `results/blackbox/qwen_grpo_lora_r16_reasonfield_v1/`

Submit the Slurm job:

```bash
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh
```

Smoke test on one GPU:

```bash
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh \
  train_global_limit=32 \
  validation_global_limit=32 \
  training.max_steps=1 \
  wandb.enabled=false
```

Useful overrides:

```bash
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh training.learning_rate=5e-7
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh vllm.enabled=false
```

GRPO can continue an existing compatible LoRA with `init_adapter`, and training
can be restricted by a literal dataset-name substring:

```bash
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh \
  --config-name qwen_grpo_lora_pid_varied \
  method=qwen9b_pid_varied_grpo_ep1_v1 \
  training.num_train_epochs=1.0
```

When `init_adapter` is set, the adapter is loaded with trainable PEFT weights and
the trainer does not create a second LoRA. Validation remains on the complete
validation split even when `train_dataset_name_contains` filters training.

Experimental SDPO uses `configs/qwen_sdpo_lora_pid_varied.yaml` and the separate
`run_qwen_sdpo_lora.sh` wrapper. It requires the external pinned environment at
`/scratch/s4626451/.venvs/aletheia-sdpo` because TRL 1.8 requires vLLM <=0.23,
while the main development environment intentionally retains vLLM 0.24. The
SDPO dataset adds teacher-only authoritative-label feedback while keeping the
student prompt unchanged.

The reward is label-supervised and parse-aware: parseable binary predictions are
converted to `[0, 1]`, positives are rewarded for predicting `1`, negatives for
predicting `0`, and invalid completions receive no correctness reward. A small
format reward encourages a visible reasoning field followed by the strict binary
prediction:

```text
<reasoning>
Concise evidence comparison.
</reasoning>
Prediction:N
```

There is currently no completion-length penalty reward. The active reward
weights are correctness `1.0` and format `0.05`. Future binary runs use
`scoring.strict_output_format=true` and `scoring.output_separator=""`, so both
correctness and format rewards require the no-space `Prediction:<0 or 1>`
contract. This aligns training with next-token logits inference: append
`Prediction:` and score token `"0"` versus token `"1"`.

## Results and Backend Notes

Binary-output run:

- Method: `qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2`
- Adapter: `results/blackbox/qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2/adapter`
- Training setting: `training.muon_learning_rate=3e-5`, `training.learning_rate=1e-6`,
  `training.temperature=1.2`, binary `Prediction: <0 or 1>` output, and no
  length penalty reward.

On validation after training, threshold `0.5` scored balanced accuracy/AUROC
`0.9262`, recall `0.9238`, FPR `0.0714`, with 1 parse error. On the local test
split with the fair vLLM LoRA evaluator, threshold `0.5` scored balanced
accuracy/AUROC `0.8762`, recall `0.7714`, FPR `0.0190`, with 6 parse errors and
`40.7s` scoring-only time (`20.2 rows/s`). This is a small balanced-accuracy
improvement over the earlier rating-field GRPO adapter while shifting the error
profile toward lower FPR and lower recall.

The binary run is therefore the strongest GRPO LoRA row so far, but it is not
competitive with the best prompt-only approaches on the internal test
leaderboard. It remains below the fast no-thinking prompt ensemble
`qwen_reason_nothink_truth_hidden_ensemble_v1` (`0.8821` balanced accuracy in
`29.7s`), the original reasoning prompt `qwen_reason_v1` (`0.9095` balanced
accuracy), and the 4096-token three-prompt reasoning ensemble (`0.9298`
balanced accuracy). Treat this run as evidence that RL can learn a clean binary,
low-FPR judge, not as evidence that the current RL setup has surpassed prompting.

Exact next-token binary-logit scoring is much stronger than generation parsing
for the same binary adapter. The evaluator
`experiments/qwen_grpo_lora/evaluate_qwen_grpo_lora_logits.py` appends an
assistant completion prefix with an empty reasoning field,
`<reasoning>\n</reasoning>\nPrediction: `, then computes
`softmax(logit("0"), logit("1"))` for the next token. It keeps the original
training prompt's `<assistant_reasoning>` input field. On the local test split,
threshold `0.5` scored balanced accuracy `0.9024`, AUROC `0.9646`, recall
`0.8714`, and FPR `0.0667` in `51.0s` scoring time. A test-set threshold sweep
peaked at balanced accuracy `0.9226` with threshold `0.3486`, recall `0.9238`,
and FPR `0.0786`. Selecting the threshold on validation instead chose `0.5622`;
applying that to test scored balanced accuracy `0.9012`, recall `0.8619`, and
FPR `0.0595`. The logits are not completely saturated: on test, the median score
for negatives was `0.0025`, the median for positives was `0.9903`, but positive
5th percentile was still `0.0675` and negative 95th percentile was `0.7311`.

Important correction: this early evaluation retained the saved
`judge.include_reasoning=true` setting and therefore supplied the dataset's
source `assistant_reasoning` field. Leaderboard inputs do not contain that
field, so do not use its `0.9646` test AUROC as a submission estimate. Pass
`--exclude-reasoning` for leaderboard-aligned evaluations.

Using populated generated reasoning before the label was worse as a calibrated
score despite looking strong on validation. With cached generated completions
stripped at `Prediction:` and then rescored with next-token logits, validation at
threshold `0.5` scored balanced accuracy `0.9262`, AUROC `0.9496`, recall
`0.9238`, and FPR `0.0714`; validation threshold selection chose a near-zero
threshold `0.000203`. On test, threshold `0.5` scored balanced accuracy
`0.8810`, AUROC `0.8686`, recall `0.7810`, and FPR `0.0190`; applying the
validation-selected threshold to test dropped balanced accuracy to `0.7655`
because FPR rose to `0.2952`. Treat generated-reasoning-then-logits as a
two-pass diagnostic, not the preferred inference route. The empty-reasoning
prefix is better calibrated and avoids paying for generation.

Strongest completed training run before the binary-output change:

- Method: `qwen_grpo_lora_r16_reasonfield_muonlr3e5_full_v1`
- Adapter: `results/blackbox/qwen_grpo_lora_r16_reasonfield_muonlr3e5_full_v1/adapter`
- Sweep setting: `training.muon_learning_rate=3e-5`, `training.learning_rate=1e-6`

Validation after training was strong: balanced accuracy/AUROC `0.9250`, recall
`0.8929`, FPR `0.0429`, zero parse errors, and about `102s` scoring time. The
model mostly learned a very short classifier-style completion:

```text
<reasoning>0</reasoning>Rating:1
<reasoning>7</reasoning>Rating:7
```

On the local test split, be careful about evaluator backend:

- The first test evaluation used Transformers/PEFT `model.generate` with
  `max_new_tokens=32`. It scored balanced accuracy `0.8988`, AUROC `0.8988`,
  recall `0.8667`, FPR `0.0690`, zero parse errors, and `101.7s` scoring time.
  This is useful as a diagnostic for the saved adapter, but it is not comparable
  to vLLM/OpenAI leaderboard rows.
- The fair vLLM LoRA evaluation used the adapter tokenizer, `max_tokens=512`,
  `temperature=0.0`, and scoring-only timing. Slurm job `30032964` completed in
  `2m31s` wall time and scored balanced accuracy `0.8738`, AUROC `0.8758`,
  recall `0.7762`, FPR `0.0286`, 6 parse errors, and `38.5s` scoring time
  (`21.3 rows/s`). This row replaced the old Transformers-timed GRPO row in
  `results/blackbox/leaderboard.md`.

The vLLM score drop is not mainly from parse errors. The six parse errors were
long completions that hit the 512-token cap before emitting a final `Rating:`;
five were positives and one was negative. Oracle-fixing only those parse errors
would raise balanced accuracy only to about `0.879`. The broader difference is
that vLLM does not reproduce the collapsed two-output behavior from the
Transformers/PEFT path: the vLLM run produced long, mostly unique reasoning
completions. Treat this adapter as backend-sensitive unless a dedicated
prompt-level HF-vs-vLLM equivalence check proves otherwise.

### Phoenix/PID continuation with output-only logits

The one- and two-epoch varied-only GRPO continuations from
`qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1` were
re-evaluated without the source `assistant_reasoning` field. The evaluator used
one direct next-token pass at the empty reasoning scaffold and normalized
literal `0`/`1` logits. Validation job `30304158` selected epoch 1 by macro
AUROC:

| adapter | macro AUROC | instructed | varied | unique scores |
| --- | ---: | ---: | ---: | ---: |
| epoch 1 | **`0.95137`** | **`0.99656`** | `0.89111` | `79/822` |
| epoch 2 | `0.94935` | `0.98667` | **`0.89958`** | `69/822` |

The epoch-1 validation threshold was `0.37754068`. Frozen local-test job
`30304374` scored epoch 1 at `0.95982` macro AUROC (`0.99073` instructed,
`0.91861` varied), with `79/821` unique scores and `38.85s` scoring time.
Balanced accuracy was `0.88690` at threshold `0.5` and `0.89881` at the frozen
validation threshold. This is a leaderboard-compatible inference path: GRPO
used privileged reasoning during training, but inference uses only the
observable output and does not generate a chain. Treat epoch 1 as the promotion
candidate and do not test epoch 2.

### Optimized Qwen-397B checkpoint continuation

The Qwen-397B soft-distillation winner
`qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1` is the initialization for the
reasoning-GRPO campaign. GRPO sees output-only varied-deception rows and is
explicitly instructed to emit a concise `<reasoning>...</reasoning>` analysis
before `Prediction:0|1`; it never receives source `assistant_reasoning`.
Promotion scoring returns to the exact original binary-only Phoenix prompt and
normalizes constrained literal next-token `0`/`1` logits without generating the
learned reasoning.

The frozen pre-GRPO vLLM reference scored validation macro AUROC `0.95369`
(`0.90833` balanced accuracy, `0.85476` recall, `0.03810` FPR at threshold
`0.5`) with `768/822` unique scores. The one-token pass took `10.23s`
(`80.36` rows/s). Its small difference from the earlier Transformers-path
`0.95393` AUROC is backend-level numeric variation; compare all GRPO candidates
against the vLLM `0.95369` reference.

Matched 16-step H100 SXM5 speed probes selected generation batch 32. A cold
batch-32 run took `230.5s` because it paid one-time Triton/TileLang autotuning;
the warm repeat took `139.7s` (`8.73s/step`). Generation batch 64 with a larger
vLLM allocation took `214.8s` (`13.43s/step`) after the caches were warm, making
batch 32 about 35% faster. On the 80 GB H100, vLLM memory utilization `0.25`
failed before training because its 20 GB allowance left no KV-cache blocks;
`0.35` initialized successfully and kept total process memory near 50--52 GiB.
The official `causal_conv1d` extension could not be safely enabled: the Lambda
image exposes CUDA 12.8's compiler while the installed PyTorch build targets
CUDA 13.0. Do not force-build an ABI-mismatched extension.

The full one-epoch `3e-5` Muon continuation completed 360 steps in `2863.9s`;
its preserved remote adapter SHA-256 is
`f5098fd8da0642dd8e5ff5f5bf63a9a1bddce2ecad050d3f894425d128e82384`.
Reasoning began near 140 tokens with about 14% clipping, then converged to
roughly 60 tokens with zero clipping in the final logged batch. This operational
improvement did not improve ranking:

| adapter | macro AUROC | instructed | varied | BA at 0.5 | recall | FPR | unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Optimized SFT reference | **`0.95369`** | **`0.99792`** | **`0.89472`** | **`0.90833`** | **`0.85476`** | **`0.03810`** | **`768/822`** |
| Reasoning GRPO epoch 1 | `0.94500` | `0.99385` | `0.87986` | `0.88452` | `0.83333` | `0.06429` | `582/822` |

GRPO lost `0.00869` macro AUROC, including `0.01486` on the varied units it
trained on. The score vector also became substantially less diverse. This
fails the predeclared requirement that the initial continuation be promising
before spending on a learning-rate/epoch sweep. Keep the optimized SFT adapter,
do not test-select the GRPO checkpoint, and do not infer better detection from
its cleaner reasoning format or lower training loss.

## Optimization Notes

Keep these measurements in mind before changing throughput-related defaults:

- `per_device_train_batch_size=4`, `num_generations=4`,
  `vllm.gpu_memory_utilization=0.25`, and vLLM sleep mode disabled is the best
  tested default so far.
- Larger train batches were not faster. Batch 8 increased step time enough to
  lose throughput versus batch 4; batch 2 was faster per step but lower overall
  prompt throughput.
- At `gpu_memory_utilization=0.25`, vLLM reported about `3.86 GiB` KV cache,
  `91,776` cache tokens, and about `19.9x` maximum concurrency for 4608-token
  requests. That already covers the current `4 * 4 = 16` train-generation
  concurrency.
- Rank-16 reasoning-field smoke runs used about `44-46 GiB` peak GPU memory and
  showed roughly `98%` active GPU utilization during generation/training windows,
  so simply raising vLLM memory fraction is unlikely to help unless generation
  concurrency is also increased.
- If tuning speed again, prefer short smoke matrices over full runs. Promising
  knobs are `training.generation_batch_size` and `evaluation.batch_size`; treat
  higher `vllm.gpu_memory_utilization` as a companion setting for larger
  generation batches, not as a standalone speed fix.

## Profiling Notes

W&B already records the useful TRL and GPU profiling streams when logging is
enabled. Check these before adding custom timers:

- Post-training validation metrics are logged directly to the active W&B run after
  the local validation split is scored, bypassing the Trainer callback key rewrite
  that would otherwise place custom keys under `train/*`. Look for keys under
  `validation/*` for macro selected-threshold and baseline-threshold metrics,
  parse errors, and scoring throughput; per-dataset metrics are under
  `validation_per_dataset/<dataset_slug>/*`.
- Phase timings are in the default history under
  `profiling/Time taken: MuonGRPOTrainer.*`, including `vLLM.generate`,
  `_move_model_to_vllm`, `_get_per_token_logps_and_entropies`, `compute_loss`,
  `_calculate_rewards`, `correctness_reward`, and `format_reward`.
- GPU utilization and FLOPs-related counters are in the W&B `events`/`system`
  stream under keys such as `system.gpu.0.gpu`, `system.gpu.0.smActive`,
  `system.gpu.0.smOccupancy`, `system.gpu.0.pipeTensorActive`,
  `system.gpu.0.pipeTensorHmmaActive`, `system.gpu.0.pipeFp16Active`,
  `system.gpu.0.dramActive`, `system.gpu.0.memoryAllocated`, and
  `system.gpu.0.powerWatts`.
- In the earlier batch-4 direct-answer run `xg9qzyd9`, W&B showed
  `vLLM.generate` mean `0.826s`, `_move_model_to_vllm` mean `0.444s`,
  `_get_per_token_logps_and_entropies` mean `0.491s`, and `compute_loss` mean
  `0.290s`. Tensor pipe activity averaged about `26%` while SM active averaged
  about `82%`; the GPU was busy but Tensor Cores were not saturated.
- The rank-16 reasoning-field smoke had W&B disabled, so we do not yet have the
  same W&B phase split for long reasoning completions. Its local logs showed
  mean completion length around `102` tokens, so generation and completion
  logprob work should matter more than in the 5-token direct-answer runs.
