# Phoenix Wright 4.4 bundled adapter

The `deception/` PEFT LoRA targets `Qwen/Qwen3.5-9B`. It is the selected
one-epoch varied-only GRPO continuation of the rank-16, alpha-32
privileged-information-distillation adapter. Phoenix 4.4 scores literal
next-token `0`/`1` logits after an empty `<reasoning>` scaffold and does not
expose source reasoning or generate a chain at inference.

NDIF repository: `Jazhyc/aletheias-phoenix-v4-4-grpo-r16`

Adapter SHA-256:
`65303d38d7681b5f5ae6953f60c4dbe2b5469e4ff0044fd60fb047b2651acd4b`

License: MIT.
