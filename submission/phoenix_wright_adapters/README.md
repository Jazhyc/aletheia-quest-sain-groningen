# Phoenix Wright adapters

All adapters bundled by the current submission live in this directory:

- `main/` is the Phoenix 6.3 rank-16 Qwen3.5-9B ordinary-row judge, trained by
  binary soft distillation from `Qwen/Qwen3.5-397B-A17B-FP8`. It matches
  `Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16` revision
  `59a9c57982c8430ff9e8bb64c3e53b8fb775dbc2`.
- `resolved_intent/` is the optional rank-1 intent specialist used only for
  eligible rows with an exposed final-assistant reasoning field.

Both bundled checkpoints use Qwen3.5's canonical multimodal PEFT tensor path,
`base_model.model.model.language_model.*`, required by the competition NDIF
client stack pinned on 2026-07-23. Do not replace them with text-only
`AutoModelForCausalLM` checkpoints whose keys omit `language_model`; see
`docs/peft_adapter_compatibility.md`.

The frozen action-report route still loads the original
`Jazhyc/aletheias-phoenix-wright-v2-adapter`; it is intentionally distinct from
the bundled Phoenix 6.3 main adapter. The explicit knowledge-report route uses
base Qwen without a LoRA.
