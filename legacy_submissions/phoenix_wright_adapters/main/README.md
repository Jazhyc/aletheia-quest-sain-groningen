---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
- deception-detection
---

# Phoenix 8 full-data Kimi K3 primary adapter

Rank-16/alpha-32 LoRA for `Qwen/Qwen3.5-9B`, trained as the ordinary-row judge
and reused for all primary routes in Phoenix 8.0. The teacher was
`moonshotai/kimi-k3`, served by Fireworks through OpenRouter and evaluated at
the literal `0|1` boundary after `Prediction:` with the frozen no-thinking
Truth Value Guard prompt. The student used all 6,573 available training rows:
3,693 instructed-deception rows and 2,880 varied-deception rows. It trained for
two epochs with AdamW at `5e-5`, effective batch size 32, and binary soft-target
BCE only.

The Qwen3.5-9B base ran in BF16. Soft targets, LoRA weights, AdamW state,
binary margins, and BCE were FP32. The adapter received no generated reasoning,
hard-label loss, completion loss, or pairwise loss. Its 256 tensors use
canonical Qwen3.5 paths under `model.language_model.layers` and exclude visual
modules.

Phoenix 8.0 applies the exact Phoenix 4.0 observable-input renderer on ordinary
rows and the semantic HP-KR/action prompts on their routed rows. Every prompt
terminates directly at `Prediction:` without a reasoning-summary instruction
or scaffold.
It scores the normalized literal `0|1` next-token margin directly, without
generating judge reasoning. Frozen ordinary local validation macro AUROC was
`0.96411` (`0.99854` instructed, `0.91819` varied). A later same-session test
anchor scored `0.96625` (`0.99479` instructed, `0.92819` varied); these are
local competition diagnostics and do not guarantee private transfer.

The matching public repository is
`Jazhyc/aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2`. The canonical
`adapter_model.safetensors` SHA-256 is
`c3be0b58b5caf5750b3dea06b5a1490cb735483adaba51f6f09568054531edc0`.
This adapter is intended for the matching competition notebook and is not a
general-purpose truth classifier.
