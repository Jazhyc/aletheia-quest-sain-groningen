# NDIF persistent-id mismatch (Qwen3.5, Jul 2026)

## Symptom

Every NDIF remote trace against Qwen3.5-27B fails immediately with:

```
RemoteException: Unknown persistent id: Module:model.model.embed_tokens
```

All Qwen datasets score 0.5 (baseline).  Gemma and Nemotron are unaffected.

## Root cause

The NDIF server redeployed Qwen3.5 with a newer transformers version
(`5.15.0.dev0`, per `/status`).  Two mismatches broke persistent-id
resolution:

### 1. Wrong model class

| | Local | Server |
|---|---|---|
| Automodel | `AutoModelForCausalLM` (default) | `AutoModelForCausalLM` → `Qwen3_5ForConditionalGeneration` |
| Result class | `Qwen3_5ForCausalLM` | `Qwen3_5ForConditionalGeneration` |
| Text layers path | `model.model.layers` | `model.model.language_model.layers` |
| Embedding path | `model.model.embed_tokens` | `model.model.language_model.embed_tokens` |

In transformers 5.15, `Qwen3_5ForCausalLM` was removed — the causal-LM
automodel now routes through the multimodal class, which nests the text
model under a `language_model` hop.  Our older transformers resolves
`AutoModelForCausalLM` to the flat `Qwen3_5ForCausalLM` class, so the
local meta-model tree doesn't match the server's.

### 2. Extra `act` submodule

Even with the correct class, `Qwen3_5GatedDeltaNet` differs:

| | Local (5.14.1) | Server (5.15.0.dev0) |
|---|---|---|
| `act` submodule | `SiLUActivation` child module | **none** (SiLU is inlined in `forward`) |
| Persistent id sent | `…linear_attn.act` | not recognised |

The server expects the `act` submodule not to exist; our local model
ships it as a persistent id and the server rejects it.

## Diagnosis steps

1. **Confirmed Qwen-specific.**  Traced Gemma → OK.  Traced Qwen → `model.model.embed_tokens` unknown.

2. **Checked NDIF `/status`.**  Qwen3.5-27B deployed as `LanguageModel` with `architectures: ['Qwen3_5ForConditionalGeneration']` — the multimodal class, not the causal-LM class.

3. **Forced `automodel=AutoModelForImageTextToText`.**  Got past the `embed_tokens` error but hit a new one: `…linear_attn.act` unknown.

4. **Inspected `linear_attn` submodules locally.**  Found `act: SiLUActivation` — confirmed it's the sole remaining mismatch.

5. **Monkey-patched `Qwen3_5GatedDeltaNet.__init__`** to strip `act` from `_modules`.  Removed the module from the envoy tree → trace succeeded.

## Fix

In `submission/util.py`:

1. **Monkey-patch at import time** — strip `act` from `Qwen3_5GatedDeltaNet._modules` before any `LanguageModel` is built.  Wrapped in `try/except` so it's a no-op on transformers versions that don't have the class.

2. **Use `automodel=AutoModelForImageTextToText`** for any model whose id contains `"qwen"`.  This loads `Qwen3_5ForConditionalGeneration` so `decoder_layers` walks through `model.model.language_model.layers` (matching the server).

Only `util.build_model` is touched.  Notebooks need no changes.

## Why the patch is safe

- The meta model is **never forwarded** during remote tracing.  Stripping `act` only affects the persistent-id catalogue sent to the server.
- The actual forward pass runs on the server with the server's own module tree.
- Other families (Gemma, Nemotron) already use the multimodal class natively and are untouched by the `"qwen"` guard.
