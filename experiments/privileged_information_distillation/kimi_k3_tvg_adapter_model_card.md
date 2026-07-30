---
base_model: Qwen/Qwen3.5-9B
library_name: peft
license: mit
pipeline_tag: text-generation
tags:
  - peft
  - lora
  - aletheias-quest
  - deception-detection
  - distillation
---

# Phoenix v7 Kimi K3 TVG soft-distillation adapter

Rank-16/alpha-32 LoRA for `Qwen/Qwen3.5-9B`, trained for continuous
direct-boundary deception scoring in Aletheia's Quest.

The teacher was `moonshotai/kimi-k3`, served by Fireworks through OpenRouter.
It scored normalized literal `0|1` log probabilities immediately after
`Prediction:` with the frozen no-thinking Truth Value Guard prompt. The student
used all 2,880 varied-deception training rows for two epochs with AdamW at
`5e-5`, effective batch size 32, and binary soft-target BCE. It received no
generated teacher reasoning, hard-label loss, completion loss, or pairwise
loss.

The frozen `Qwen/Qwen3.5-9B` base ran in BF16. Soft targets, LoRA weights,
AdamW state, binary margins, and BCE were FP32. The 256 adapter tensors use
canonical Qwen3.5 paths under `model.language_model.layers` and explicitly
exclude visual modules.

Frozen local-validation direct-margin results:

| metric | value |
| --- | ---: |
| Macro AUROC | 0.95994 |
| Instructed AUROC | 0.99833 |
| Varied AUROC | 0.90875 |
| Balanced accuracy at 0.5 | 0.91548 |

These are local competition diagnostics and do not guarantee private-dataset
transfer. The adapter is intended for continuous literal `0|1` boundary
scoring; it does not generate judge reasoning at inference.
