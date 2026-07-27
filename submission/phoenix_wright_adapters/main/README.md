---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
---

# Phoenix Wright main adapter

Rank-16, alpha-32 Qwen3.5-9B LoRA trained for one epoch on all 2,880
varied-deception rows using concise privileged Truth Value Guard reasoning
traces produced by GPT-5.6 Luna at medium reasoning effort. The teacher received
the authoritative label while constructing each trace. Training used AdamW at
`5e-5` with effective batch size 32.

Competition consumers render observable conversation and output only and
disable Qwen thinking. They can normalize literal `0|1` next-token logits
either directly after `Prediction:` or after generating the learned rationale
and removing its selected label. The adapter is intended for the matching
competition notebooks and is not a general-purpose truth classifier.

The `adapter_model.safetensors` SHA-256 is
`da83f294a248a97ddf07ff7a458280f0990dea329e61fc5e2aec8faec8c61d17`.
The deployed Hugging Face repository is
`Jazhyc/aletheias-phoenix-luna-medium-reasoning-r16`; the initial upload commit
is `fa52f5e7f6a36d7d06e6718c65bd00fd4d11658f`.
