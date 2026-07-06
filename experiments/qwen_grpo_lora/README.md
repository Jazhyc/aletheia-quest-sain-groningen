# Qwen No-Thinking GRPO LoRA

This experiment fine-tunes the local `Qwen/Qwen3.5-9B` no-thinking judge with
TRL GRPO and a rank-16 LoRA adapter. It trains only on `dev_splits/dry.train.yaml`
and uses `dev_splits/dry.validation.yaml` for post-train validation metrics and
threshold selection.

Default setup:

- Base model: `Qwen/Qwen3.5-9B`
- Prompt: no-thinking judge prompt with a visible `<reasoning>` field and final rating
- Chat template: `enable_thinking=false`
- LoRA: rank `16`, alpha `32`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Training: one epoch, default TRL DAPO objective, Muon over LoRA matrices,
  512-token completions, colocated vLLM generation, W&B entity
  `aletheia-quest`, project `aletheias-quest-qwen-grpo`
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

The reward is label-supervised and parse-aware: parseable ratings are converted
to `[0, 1]`, positives are rewarded for high scores, negatives for low scores,
and invalid completions receive no correctness reward. A small format reward
encourages a visible reasoning field followed by the strict rating:

```text
<reasoning>
Concise evidence comparison.
</reasoning>
Rating: N
```

A matching 5% length penalty uses generated token count normalized by the
512-token completion budget, so using the full budget contributes `-0.05` after
reward weighting.

## Results and Backend Notes

Current strongest training run:

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
