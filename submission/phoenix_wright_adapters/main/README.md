---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
---

# Phoenix Wright 5.1 main adapter

Rank-16, alpha-32 Qwen3.5-9B LoRA trained for one epoch on 2,875 parsed
varied-deception reasoning traces produced by GPT-OSS-120B. The teacher was
ground-truth blind: it saw only the observable conversation and final output,
generated a concise material-claim rationale, and supplied its own binary
decision. Its mistakes were retained rather than corrected or filtered by
ground-truth labels. Training used AdamW at `5e-5` with effective batch size 32.

The submission renders observable conversation and output only, disables Qwen
thinking, and normalizes the next-token logits for literal `0|1` directly after
`Prediction:` without generating the learned rationale. The adapter is intended
for the matching competition notebook and is not a general-purpose truth
classifier.

The `adapter_model.safetensors` SHA-256 is
`bfe4a35ceb11055e7b7e77d29cc7027ccb5daa355703d9acd4d14eb03db01931`.
The deployed Hugging Face repository is
`Jazhyc/aletheias-phoenix-blind-reasoning-r16`; the initial upload commit is
`eaeb37ce03d791db340114eb9132494d7081d31f`.
