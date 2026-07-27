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

The 26 resumable competition-NDIF sessions took 185.6 seconds of scoring time.
Gemma passed local tokenizer and prompt compatibility (`0` and `1` are distinct
single tokens), but the competition endpoint rejected two separate two-row
smoke requests before model execution because `google/gemma-3-27b-it` was not
pinned and this API key does not support hotswapping. The regular NDIF endpoint,
`https://api.ndif.us`, listed Gemma as a hot Pilot-only deployment, but a
separate approved two-row access check rejected the competition key as missing
or invalid before accepting the trace. Therefore neither endpoint can currently
run Gemma with the available key. This is an endpoint/access blocker, not a
benchmark result; no local model weights are required.
