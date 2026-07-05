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
- Training: one epoch, Muon over LoRA matrices, 512-token completions,
  colocated vLLM generation, W&B entity `aletheia-quest`, project
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
