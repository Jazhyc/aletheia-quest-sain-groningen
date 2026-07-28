---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
---

# Phoenix Wright 5.3 main adapter

Rank-16, alpha-32 Qwen3.5-9B LoRA selected by the original
privileged-information-distillation validation sweep. GPT-OSS-120B supplied
concise Truth Value Guard reasoning summaries conditioned on the authoritative
labels. This is the checkpoint served by
`Jazhyc/aletheias-phoenix-wright-v2-adapter`, which the Phoenix 4.0 notebook
actually loaded during remote leaderboard execution.

Competition consumers render observable conversation and output only and
disable Qwen thinking. Phoenix 5.3 normalizes literal `0|1` next-token logits
directly after `Prediction:`. The adapter is intended for the matching
competition notebook and is not a general-purpose truth classifier.

The legacy text-only checkpoint SHA-256 was
`1dff12c68935117da59f9969c2b26141d294f0ffb1be7dfe0193cfd5863c2b29`.
The bundled canonical-path serialization is
`933ba79505cd1b73e69c5df05ed87c12246da355581e03f0b1a1bb7295cff494`;
only PEFT tensor names changed. The hosted canonical-path migration commit is
`112b0a02a31089e51bd333d4f4c5036316cb0e2b`.
