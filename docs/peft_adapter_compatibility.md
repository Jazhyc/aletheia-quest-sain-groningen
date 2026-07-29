# Qwen3.5 PEFT compatibility with competition NDIF

## 2026-07-28 incident

The competition NDIF client update pinned on 2026-07-23 changed the default
Qwen3.5 `LanguageModel` meta-model to the repository's canonical multimodal
loader. Qwen3.5 is registered with `AutoModelForImageTextToText`, whose decoder
lives under:

```text
model.language_model.layers.*
```

Our Phoenix LoRAs were trained with `AutoModelForCausalLM`, whose text-only
decoder lives under:

```text
model.layers.*
```

The corresponding saved PEFT tensor prefixes were:

```text
text-only training:  base_model.model.model.layers.*
canonical NDIF:      base_model.model.model.language_model.layers.*
```

This is a model-tree compatibility failure, not a change to the public PEFT
loading API. `PeftModel.from_pretrained(base_model, adapter_path)` and
`LanguageModel(model_id, peft=adapter_id)` remain the correct calls.

## Evidence

The current NNsight wheel explicitly promotes multimodal repository configs to
`AutoModelForImageTextToText`. Its rationale warns that the text-only tree can
cause canonically trained PEFT checkpoints to match zero parameter names.

Our competition-NDIF fingerprint confirmed the inverse mismatch in practice.
Three independently trained Phoenix adapters produced bit-identical float32
scores to base Qwen3.5-9B on the same 32 prompts. Their shared score digest was
`f6ffa66cb58bf68289eef610762afadcc4ad05f07a495f8f1d0a8e74db949fa5`.
An official Qwen3.5-27B competition organism changed the matched base scores,
confirming that NDIF's PEFT mechanism itself remained active.

Ignored diagnostic artifacts are under:

```text
results/blackbox/phoenix_adapter_ndif_fingerprint_v1/
```

## Remediation

Existing text-only checkpoints can be migrated without retraining by inserting
`language_model` into every saved decoder tensor key. Tensor values, shapes,
dtypes, and LoRA hyperparameters must remain unchanged. The adapter config
should also exclude visual modules:

```text
.*(visual|vision_tower|merger|patch_embed).*
```

Use:

```bash
python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py
python experiments/heterogeneous_adapter_ensemble/migrate_qwen35_peft_paths.py --upload
```

The first command downloads and verifies every Aletheia adapter owned by the
authenticated Hugging Face account without changing remote state. The second
creates recoverable Hugging Face commits with optimistic parent-revision
checks. Its ignored manifest records old/new revisions and file hashes.

For future Qwen3.5 LoRAs intended for competition NDIF, train against
`AutoModelForImageTextToText` and explicitly exclude the vision tower. If a
text-only training model remains necessary for memory or trainer
compatibility, treat checkpoint path migration plus a base-versus-adapter NDIF
fingerprint as mandatory deployment steps.

## Completed hosted migration

All ten Aletheia/Phoenix adapters owned by `Jazhyc` were migrated on
2026-07-28. Hugging Face commit history retains the pre-migration revisions.

| adapter | old revision | canonical-path revision |
| --- | --- | --- |
| `aletheias-phoenix-blind-reasoning-r16` | `37adee41ed85b218b6e04803ba23f84205d05a88` | `955c81648632ffe749f32d5c2f9fefb8b4f15297` |
| `aletheias-phoenix-luna-medium-reasoning-r16` | `efbb2cf750b8e60ab789ea369e797fb7c6cbbdf2` | `f5d4f782ff83e0c77cea9b7374d31ce098120cfe` |
| `aletheias-phoenix-v3-deception-r1` | `bd8d57dfe8f711d77f4ebcef04d95c6d2ef21d52` | `4577cc83c3e6fb64e09b383f52f92226227e2990` |
| `aletheias-phoenix-v3-incorrectness-r1` | `068bedcf4b93e325e19ad78fbf5ca36351243118` | `e96b4b99d8dfa4f6a79d2b3aeffae07649e37034` |
| `aletheias-phoenix-v3-resolved-intent-r1` | `2ccd1f5abaac4495ad1683bad77c9c710c99bce0` | `7c23558e34f7fa1fe7d06e5bf28ffbb54b0483a2` |
| `aletheias-phoenix-v4-4-grpo-r16` | `9c1ea176007d38fd80148dda9289143e53d2adcf` | `6be3c41f0ac496dd1ddea872259b85d6b2734565` |
| `aletheias-phoenix-v5-qwen27-tvg-soft-r16` | `274c8dad9b308c555bfc672d2e5108be8fd00cc7` | `0a11d1f349825063f0f5466ec9f775314b69fcf8` |
| `aletheias-phoenix-wright-v2-1-adapter` | `2134cac453b3f9996a4668b1ff10cb3d629e00fa` | `edd4c12c67a1dd2751dc081855792f5e63e2a544` |
| `aletheias-phoenix-wright-v2-adapter` | `cb1d515230fc2f6615287ca75ee4650059b0ea28` | `112b0a02a31089e51bd333d4f4c5036316cb0e2b` |
| `aletheias-phoenix-wright-v3.1-rank24-adapter` | `3c9d7b129df64c4dccfe0242ec407df2ddac0d21` | `b07c0e297ab8294bd7a4ac3fd2baa89683362f69` |

Every hosted LFS weight digest matched the locally verified converted file
after upload. The submission-bundled Luna rank-16 and resolved-intent rank-1
adapters were migrated to the same canonical layout.

Phoenix 6.3's Qwen397 soft-distillation adapter was born through the guarded
text-only-training-plus-canonical-migration path and published directly with
canonical keys. Repository
`Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16`, revision
`59a9c57982c8430ff9e8bb64c3e53b8fb775dbc2`, has LFS weight SHA-256
`fac7b031bf52bc5597ee02a6dec20f355b3c3297bf110d3bb0233f629a69004d`.

## Luna 5.2 confirmation

The first 32-row competition-NDIF canary used identical prompts for base Qwen
and the converted Luna adapter. No score was exactly or approximately equal:
mean absolute difference was `0.19303`, maximum absolute difference was
`0.38138`, and Pearson correlation was `0.71709`. This rejected the earlier
silent-base behavior.

A direct full-unit confirmation then scored all 400 rows of
`dev-varied-deception-Qwen3.5-27B-None` at batch 32:

| metric | converted Luna 5.2 |
| --- | ---: |
| AUROC | `0.967075` |
| balanced accuracy at `0.5` | `0.9125` |
| unique scores | `54 / 400` |
| score range | `0.06009`–`0.98201` |
| NDIF scoring time | `29.13s` |

The full score digest was
`eec505b47956755185748eba462eb21c70667556ada03228983913c38610daa8`.
This is a deployment confirmation on one public unit, not a new
validation-selected promotion result.

## Corrected four-adapter validation

A complete competition-NDIF rerun then evaluated the migrated GPT-OSS
privileged, GPT-OSS blind, Luna privileged, and Qwen3.5-27B soft-distillation
adapters on all 822 validation rows. It used Phoenix's optimized direct-logit
path: one session per model, length-aware `48/32/16` batches,
`logits_to_keep=1`, no generation, and normalized literal `0|1` scores.

| adapter | training prompt | corrected macro AUROC | old vLLM macro AUROC |
| --- | --- | ---: | ---: |
| GPT-OSS PI | summary | `0.94911` | `0.93815` |
| GPT-OSS blind | summary | `0.94893` | `0.93762` |
| Luna PI | summary | `0.94821` | `0.93762` |
| Q27 soft | binary | **`0.95375`** | `0.94625` |

The historical score vectors correlate at `0.99915`--`0.99936` with corrected
base-Qwen scores under the same prompts, demonstrating that they were
predominantly base-model measurements. The corrected adapters differ
materially from base. See
`docs/privileged_information_distillation/findings.md` and
`experiments/phoenix_adapter_validation_ndif/` for prompt crosses, activation
statistics, and per-dataset AUROCs.
