# Phoenix Wright 8.1 adapter

The adapter bundled by the current submission lives in this directory:

- `main/` is the rank-16 Qwen3.5-9B judge trained on the Phoenix 8 competition
  rows plus four Liars' Bench response modes by binary soft distillation from
  `moonshotai/kimi-k3`. Phoenix 8.1 uses it for
  ordinary, HP-KR, and action-report rows with their respective frozen prompts.
  It matches
  `Jazhyc/aletheias-phoenix-v8-1-kimi-k3-liars-full-r16-ep2`.

The bundled checkpoint uses Qwen3.5's canonical multimodal PEFT tensor path,
`base_model.model.model.language_model.*`, required by the competition NDIF
client stack pinned on 2026-07-23. Do not replace it with a text-only
`AutoModelForCausalLM` checkpoint whose keys omit `language_model`; see
`docs/peft_adapter_compatibility.md`.

Ordinary, HP-KR, and action-report prompts are scored together in one model
session. Phoenix 8.1 retains Phoenix 8.0's removal of the rank-1
resolved-intent adapter and score blend.
