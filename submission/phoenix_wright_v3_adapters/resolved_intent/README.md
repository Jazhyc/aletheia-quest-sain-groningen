---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
---

# Phoenix Wright 3.0 — resolved deceptive intent

Rank-1, alpha-2 LoRA used as the resolved-intent member of the frozen Phoenix
Wright 3.0 heterogeneous judge. It is queried only when an assistant reasoning
trace is present and emits `Prediction:0` or `Prediction:1` after a concise
assessment. This adapter is intended for the matching competition notebook.
