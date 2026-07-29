# Phoenix Wright 7.0 adapters

All adapters bundled by the current submission live in this directory:

- `main/` is the Phoenix 7.0 rank-16 Qwen3.5-9B judge, trained by binary soft
  distillation from `Qwen/Qwen3.5-397B-A17B-FP8`. Phoenix 7.0 uses it for
  ordinary, HP-KR, and action-report rows with their respective frozen prompts.
  It matches
  `Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2`.
- `resolved_intent/` is the optional rank-1 intent specialist used only for
  eligible rows with an exposed final-assistant reasoning field.

Both bundled checkpoints use Qwen3.5's canonical multimodal PEFT tensor path,
`base_model.model.model.language_model.*`, required by the competition NDIF
client stack pinned on 2026-07-23. Do not replace them with text-only
`AutoModelForCausalLM` checkpoints whose keys omit `language_model`; see
`docs/peft_adapter_compatibility.md`.

The HP-KR and action-report routes no longer load separate model sessions.
Only the rank-1 intent route uses an additional adapter.
