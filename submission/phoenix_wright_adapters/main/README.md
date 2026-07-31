---
license: mit
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- aletheias-quest
- lora
- deception-detection
---

# Phoenix 8.1 Kimi K3 plus Liars primary adapter

Rank-16/alpha-32 LoRA for `Qwen/Qwen3.5-9B`, trained as the ordinary-row judge
and reused for all primary routes in Phoenix 8.1. The student combines all
6,573 Phoenix 8 rows with 6,576 label-balanced Liars' Bench rows spanning
harm-pressure choice, harm-pressure knowledge reports, insider trading, and
soft triggers. Kimi K3 supplied binary soft targets through OpenRouter. It
trained for two epochs with AdamW at `5e-5`, effective batch size 32, and
binary soft-target BCE only.

The Qwen3.5-9B base ran in BF16. Soft targets, LoRA weights, AdamW state,
binary margins, and BCE were FP32. The adapter received no generated reasoning,
hard-label loss, completion loss, or pairwise loss. Its 256 tensors use
canonical Qwen3.5 paths under `model.language_model.layers` and exclude visual
modules.

Phoenix 8.1 applies the exact Phoenix 4.0 observable-input renderer on ordinary
rows and the semantic HP-KR/action prompts on their routed rows. Every prompt
terminates directly at `Prediction:` without a reasoning-summary instruction
or scaffold.
It scores the normalized literal `0|1` next-token margin directly without
generating judge reasoning. On the disjoint 800-row Liars holdout,
macro-category AUROC increased from `0.86255` for Phoenix 8 to `0.93915`.
Matched competition validation macro AUROC was `0.96214`, versus `0.96417` for
Phoenix 8, making this an explicit OOD-transfer choice rather than an
in-distribution validation promotion.

The matching public repository is
`Jazhyc/aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2`. The canonical FP32
`adapter_model.safetensors` SHA-256 is
`7159a413cf7bf569b1e7819f17b54248d48b8e18b8d56be950b872445195e136`.
A BF16 canary was not exact for these weights, so Phoenix 8.1 retains the FP32
master for deployment.
This adapter is intended for the matching competition notebook and is not a
general-purpose truth classifier.

The matching MIT Kimi binary-logit annotation artifact and exact prompt
renderer are published at
`Jazhyc/aletheias-phoenix-v8-1-kimi-k3-distillation` (Hugging Face dataset).
