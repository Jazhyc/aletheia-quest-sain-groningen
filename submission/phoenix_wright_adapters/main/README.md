---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
- deception-detection
---

# Phoenix Wright 7.0 primary adapter

Rank-16/alpha-32 LoRA for `Qwen/Qwen3.5-9B`, trained as the ordinary-row judge
in Phoenix Wright 6.3 and reused for all primary routes in Phoenix 7.0. The teacher was
`Qwen/Qwen3.5-397B-A17B-FP8`, evaluated at the literal `0|1` boundary after
`Prediction:` with the frozen no-thinking Truth Value Guard prompt. The student
used all 2,880 varied-deception training rows for two epochs with AdamW at
`5e-5`, effective batch size 32, and binary soft-target BCE only.

The Qwen3.5-9B base ran in BF16. Soft targets, LoRA weights, AdamW state,
binary margins, and BCE were FP32. The adapter received no generated reasoning,
hard-label loss, completion loss, or pairwise loss. Its 256 tensors use
canonical Qwen3.5 paths under `model.language_model.layers` and exclude visual
modules.

Phoenix 7.0 applies the exact Phoenix 4.0 observable-input renderer on ordinary
rows and the frozen HP-KR/action empty-scaffold prompts on their routed rows.
It scores the normalized literal `0|1` next-token margin directly, without
generating judge reasoning. Frozen ordinary local validation macro AUROC was `0.95393`
(`0.99833` instructed, `0.89472` varied), with 665 unique scores over 822 rows.

The matching public repository is
`Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2` at revision
`11cd26e44b77cf86064ed2246504952681d25695`. The canonical
`adapter_model.safetensors` SHA-256 is
`da32c67a2ea0d9834fb90b93dac84a9ee68f05b2a2023b8ea272704fdb58c40d`.
This adapter is intended for the matching competition notebook and is not a
general-purpose truth classifier.
