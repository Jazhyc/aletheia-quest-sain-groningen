---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
pipeline_tag: text-generation
tags:
- aletheias-quest
- deception-detection
- lora
- qwen3.5
---

# Phoenix Wright 3.1 rank-24 deception adapter

This is the rank-24, alpha-48 `Qwen/Qwen3.5-9B` LoRA used by the Phoenix Wright
3.1 Aletheia's Quest submission. It was trained on all 2,877 parsed,
label-consistent varied-deception privileged reasoning summaries for one epoch
with AdamW at `5e-5` and effective batch size 32. It targets the attention and
MLP projection modules and emits a concise evidence summary followed by
`Prediction:0` or `Prediction:1`.

Matched local validation against the rank-16 predecessor showed a small gain,
but all available varied-deception training rows are Qwen-family. This adapter
should therefore not be treated as evidence of cross-family generalization or
as a general-purpose truth classifier.

The model is published for reproducibility of the matching competition
notebook. License: MIT.
