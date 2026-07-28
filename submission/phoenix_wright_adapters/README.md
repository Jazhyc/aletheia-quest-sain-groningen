# Phoenix Wright adapters

All adapters bundled by the current submission live in this directory:

- `main/` is the rank-16 Qwen3.5-9B GPT-OSS privileged-information student
  actually loaded remotely by Phoenix 4.0 and restored for the Phoenix 5.3
  leaderboard-consistency probe.
- `resolved_intent/` is the optional rank-1 intent specialist used only for
  eligible rows with an exposed final-assistant reasoning field.

Both bundled checkpoints use Qwen3.5's canonical multimodal PEFT tensor path,
`base_model.model.model.language_model.*`, required by the competition NDIF
client stack pinned on 2026-07-23. Do not replace them with text-only
`AutoModelForCausalLM` checkpoints whose keys omit `language_model`; see
`docs/peft_adapter_compatibility.md`.

The frozen action-report route uses the same
`Jazhyc/aletheias-phoenix-wright-v2-adapter` as the 5.3 main route. The public
repository and bundled checkpoint both use the canonical Qwen3.5 PEFT tensor
path; their tensor values are equivalent to the legacy Phoenix 4.0 checkpoint.
