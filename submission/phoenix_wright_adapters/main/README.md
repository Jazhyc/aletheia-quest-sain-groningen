---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
- deception-detection
---

# Phoenix Wright primary adapter

Rank-16/alpha-32 LoRA for `Qwen/Qwen3.5-9B`, trained as the ordinary-row judge
and reused for all primary routes in Phoenix 7.0 and 8.0. The teacher was
`moonshotai/kimi-k3`, served by Fireworks through OpenRouter and evaluated at
the literal `0|1` boundary after
`Prediction:` with the frozen no-thinking Truth Value Guard prompt. The student
used all 2,880 varied-deception training rows for two epochs with AdamW at
`5e-5`, effective batch size 32, and binary soft-target BCE only.

The Qwen3.5-9B base ran in BF16. Soft targets, LoRA weights, AdamW state,
binary margins, and BCE were FP32. The adapter received no generated reasoning,
hard-label loss, completion loss, or pairwise loss. Its 256 tensors use
canonical Qwen3.5 paths under `model.language_model.layers` and exclude visual
modules.

Phoenix 8.0 applies the exact Phoenix 4.0 observable-input renderer on ordinary
rows and the frozen HP-KR/action empty-scaffold prompts on their routed rows.
It scores the normalized literal `0|1` next-token margin directly, without
generating judge reasoning. Frozen ordinary local validation macro AUROC was
`0.95994` (`0.99833` instructed, `0.90875` varied).

The matching public repository is
`Jazhyc/aletheias-phoenix-v7-kimi-k3-tvg-soft-r16-ep2` at revision
`74e5cbdf4d3c93c57f4b304c0d66cb1c46c84e92`. The canonical
`adapter_model.safetensors` SHA-256 is
`c5025a39dd05af16405c692a0c1b70657afd4f8e4a4e634bd789b5d67b4a9eb0`.
This adapter is intended for the matching competition notebook and is not a
general-purpose truth classifier.
