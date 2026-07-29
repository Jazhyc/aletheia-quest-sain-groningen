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
---

# Phoenix 6.3 Qwen397 TVG soft-distillation adapter

Rank-16/alpha-32 LoRA for `Qwen/Qwen3.5-9B`, trained as the ordinary-row judge
in the Phoenix Wright 6.3 Aletheia's Quest submission.

The teacher was `Qwen/Qwen3.5-397B-A17B-FP8`. It scored the normalized literal
`0|1` boundary immediately after `Prediction:` with the frozen no-thinking
Truth Value Guard prompt. The student used all 2,880 varied-deception training
rows for two epochs with AdamW at `5e-5`, effective batch size 32, and only
binary soft-target BCE. It received no generated reasoning, hard-label loss,
completion loss, or pairwise loss.

The frozen Qwen3.5-9B base ran in BF16. Soft targets, LoRA weights, AdamW state,
binary margins, and BCE were FP32. The 256 adapter tensors use canonical
Qwen3.5 paths under `model.language_model.layers` and explicitly exclude visual
modules.

Local validation direct-margin results:

| metric | value |
| --- | ---: |
| Macro AUROC | 0.95393 |
| Instructed AUROC | 0.99833 |
| Varied AUROC | 0.89472 |
| Balanced accuracy at 0.5 | 0.90595 |
| Unique scores | 665 / 822 |

These results are local competition diagnostics and do not guarantee transfer
to private datasets. The adapter is intended for continuous direct-boundary
scoring; it does not generate judge reasoning at inference.
