# Phoenix Wright 3.2 bundled adapter

This PEFT LoRA adapter targets `Qwen/Qwen3.5-9B`. Phoenix 3.2 uses the
full-varied-data privileged-information-distillation adapter at rank 24,
alpha 48, trained for one epoch with AdamW at `5e-5`. The submission uses the
matching public Hugging Face repository for NDIF execution and includes this
copy so the exact weights are archived with the entry.

Phoenix also bundles `submission/phoenix_wright_v3_reasoning_stack.joblib`, a
reasoning-only word n-gram plus two-feature logistic stack. It is evaluated
locally on rows with final-assistant reasoning and falls back exactly to the
adapter prediction when reasoning is absent.

License: MIT.
