# NDIF Truth Value Guard model swaps

This benchmark compares the hosted base checkpoints on the competition-specific
NDIF endpoint, `https://aletheias.api.ndif.us`,
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` and
`google/gemma-3-27b-it` as black-box deception judges on the frozen 821-row
local test split. It uses the exact compact no-thinking Truth Value Guard
prompt, the 3,000-character tail context renderer, and normalized next-token
logits for literal `0|1` directly after `Prediction:`.

The evaluator does not load model weights on the cluster. Tokenization and
metric calculation are local; model forward passes use NDIF remote sessions.
Source-model reasoning fields are never included.

Run a two-row compatibility smoke first:

```bash
bash experiments/ndif_tvg_model_swap/run_ndif_tvg_model_swap.sh \
  --limit 2 \
  --output-dir results/blackbox/ndif_tvg_model_swap_test_v1/smoke
```

Then run the complete test benchmark:

```bash
bash experiments/ndif_tvg_model_swap/run_ndif_tvg_model_swap.sh \
  --model nemotron
bash experiments/ndif_tvg_model_swap/run_ndif_tvg_model_swap.sh \
  --model gemma
```

Gemma starts at batch 8, matching the previously successful forced-logit NDIF
path. Nemotron starts at batch 1. Results are cached in row shards, validated
against dataset/index keys and prompt hashes, and safely resume after a remote
failure. Reports include macro and per-dataset AUROC, scenario AUROC, secondary
binary diagnostics, score uniqueness/ties, prompt lengths, and scoring time.

## Frozen test status (2026-07-27)

Nemotron completed all 821 rows:

| model | macro AUROC | instructed | varied | BA | recall | FPR | unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nemotron Super 120B-A12B | `0.95655` | `0.98708` | `0.91583` | `0.89762` | `0.82619` | `0.03095` | `70/821` |
| Gemma 3 27B IT | `0.92917` | `0.96344` | `0.88347` | `0.86548` | `0.85476` | `0.12381` | `136/821` |

The 26 resumable Nemotron sessions took 185.6 seconds of scoring time. Gemma
needed four sessions and 49.3 seconds at batch 8. An initial Gemma attempt
incorrectly used `VisionLanguageModel`, inferred from its Hugging Face vision
configuration. The competition deploys Gemma under the text-only
`LanguageModel` key; using the wrong wrapper missed the pinned deployment and
caused a misleading hotswap error. The corrected wrapper completed all rows.
Gemma's margins were heavily saturated: 270 rows were exactly 1.0 and only 136
unique scores remained.
