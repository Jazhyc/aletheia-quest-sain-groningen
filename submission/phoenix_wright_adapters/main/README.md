---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
---

# Phoenix Wright 5.0 — Qwen-27B TVG soft student

Fresh rank-16, alpha-32 Qwen3.5-9B LoRA trained for one epoch on the 2,880
varied-deception rows using only binary soft-target BCE. Targets are the direct
`Prediction:` probabilities of a Qwen3.5-27B Truth Value Guard teacher. Training
used AdamW at `5e-5` with effective batch size 32 and did not use hard labels,
GPT-OSS targets, or teacher/student reasoning traces.

Phoenix 5.0 renders observable conversation and output only, disables Qwen
thinking, and normalizes the next-token logits for literal `0|1` directly after
`Prediction:`. The adapter is intended for the matching competition notebook
and is not a general-purpose truth classifier.

The `adapter_model.safetensors` SHA-256 is
`c096e5839fe96fb7409fe759a79bc0989ef1efe5aee010b83c3157d38586a75f`.
The deployed Hugging Face repository is
`Jazhyc/aletheias-phoenix-v5-qwen27-tvg-soft-r16`; the initial upload commit is
`c5568edfe22fa335cbe31726ea144669e8b54314`.
