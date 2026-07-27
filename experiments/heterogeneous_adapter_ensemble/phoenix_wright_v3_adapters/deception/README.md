---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
---

# Phoenix Wright 4.4 — output-only GRPO judge

Rank-16, alpha-32 LoRA produced by one epoch of varied-only GRPO from the
selected privileged-reasoning SFT adapter. Phoenix 4.4 uses the frozen
leaderboard-aligned readout: observable conversation and output only, Qwen
thinking disabled, an empty `<reasoning>` scaffold, and normalized logits for
literal `Prediction:0|1`. It is intended for the matching competition notebook
and is not a general-purpose truth classifier.
