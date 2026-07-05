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
