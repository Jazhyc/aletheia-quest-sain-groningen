# Phoenix Wright adapters

All adapters bundled by the current submission live in this directory:

- `main/` is the rank-16 Qwen3.5-9B Luna reasoning student used for ordinary
  rows.
- `resolved_intent/` is the optional rank-1 intent specialist used only for
  eligible rows with an exposed final-assistant reasoning field.

Both bundled checkpoints use Qwen3.5's canonical multimodal PEFT tensor path,
`base_model.model.model.language_model.*`, required by the competition NDIF
client stack pinned on 2026-07-23. Do not replace them with text-only
`AutoModelForCausalLM` checkpoints whose keys omit `language_model`; see
`docs/peft_adapter_compatibility.md`.

The frozen action-report route uses
`Jazhyc/aletheias-phoenix-wright-v2-adapter` remotely. Bundling that additional
rank-16 adapter would exceed the 200 MB submission limit.
