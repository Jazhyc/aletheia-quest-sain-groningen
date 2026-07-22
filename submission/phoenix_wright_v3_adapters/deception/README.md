---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
---

# Phoenix Wright 3.0 — original Phoenix v2 judge

Rank-16, alpha-32 LoRA originally deployed by Phoenix v2.0/v2.1 and restored as
the sole judge in Phoenix v3.0. It was trained on the full varied-deception
privileged-summary recipe and emits a concise assessment followed by
`Prediction:0` or `Prediction:1`. This adapter is intended for the matching
competition notebook and is not a general-purpose truth classifier.
