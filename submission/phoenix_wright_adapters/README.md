# Phoenix Wright 5.1 adapters

All adapters bundled by the current submission live in this directory:

- `main/` is the rank-16 Qwen3.5-9B blind-reasoning student used for ordinary
  rows.
- `resolved_intent/` is the optional rank-1 intent specialist used only for
  eligible rows with an exposed final-assistant reasoning field.

The frozen action-report route uses
`Jazhyc/aletheias-phoenix-wright-v2-adapter` remotely. Bundling that additional
rank-16 adapter would exceed the 200 MB submission limit.
