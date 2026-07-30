# Phoenix Wright 8.0 adapter

The adapter bundled by the current submission lives in this directory:

- `main/` is the rank-16 Qwen3.5-9B judge trained on instructed and varied
  deception by binary soft distillation from `moonshotai/kimi-k3`. Phoenix 8.0
  uses it for
  ordinary, HP-KR, and action-report rows with their respective frozen prompts.
  It matches
  `Jazhyc/aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2`.

The bundled checkpoint uses Qwen3.5's canonical multimodal PEFT tensor path,
`base_model.model.model.language_model.*`, required by the competition NDIF
client stack pinned on 2026-07-23. Do not replace it with a text-only
`AutoModelForCausalLM` checkpoint whose keys omit `language_model`; see
`docs/peft_adapter_compatibility.md`.

Ordinary, HP-KR, and action-report prompts are scored together in one model
session. Phoenix 8.0 removes the rank-1 resolved-intent adapter and score blend.
