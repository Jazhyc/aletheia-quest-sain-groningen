# Qwen No-Thinking GRPO LoRA

This experiment fine-tunes the local `Qwen/Qwen3.5-9B` no-thinking judge with
TRL GRPO and a rank-1 LoRA adapter. It trains only on `dev_splits/dry.train.yaml`
and uses `dev_splits/dry.validation.yaml` for post-train validation metrics and
threshold selection.

Default setup:

- Base model: `Qwen/Qwen3.5-9B`
- Prompt: `qwen_reason_nothink_reasoning_output_consistency_v1`
- Chat template: `enable_thinking=false`
- LoRA: rank `1`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Training: one epoch, colocated vLLM generation, W&B entity `aletheia-quest`,
  project `aletheias-quest-qwen-grpo`
- Output: `results/blackbox/qwen_grpo_lora_r1_nothink_v1/`

Submit the Slurm job:

```bash
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh
```

Smoke test on one GPU:

```bash
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh \
  train_limit=32 \
  validation_limit=32 \
  training.max_steps=1 \
  wandb.enabled=false
```

Useful overrides:

```bash
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh training.learning_rate=5e-7
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh training.max_completion_length=512
sbatch experiments/qwen_grpo_lora/run_qwen_grpo_lora.sh vllm.enabled=false
```

The reward is label-supervised and parse-aware: parseable ratings are converted
to `[0, 1]`, positives are rewarded for high scores, negatives for low scores,
and invalid completions receive no correctness reward. A small format reward
encourages the strict `Rating: N` output.
