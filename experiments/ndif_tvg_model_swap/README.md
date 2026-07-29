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

The evaluator also exposes `--model qwen` for an unadapted Qwen3.5-27B
same-example control. This isolates whether a generating-organism LoRA, rather
than the underlying dataset unit, changes matched self-judgment.

## Matched-organism diagnostic

`run_ndif_matched_organism_tvg.py` scores every row with the exact base model
and LoRA (or no LoRA) that generated it. It groups the 21 frozen test units into
12 exact organisms to avoid repeated adapter loads, while retaining macro
per-dataset evaluation:

```bash
bash experiments/ndif_tvg_model_swap/run_ndif_matched_organism_tvg.sh
```

The model forward passes and literal `0|1` logit reads use the competition NDIF
endpoint. Local work is limited to cached-dataset loading, tokenization, and
metric calculation. Use `--dataset-regex` with `--limit-per-organism` for
compatibility smokes; use a separate output directory so smoke caches cannot be
mistaken for the complete result.

The frozen 821-row run completed on 2026-07-27:

| judge assignment | macro AUROC | instructed | varied | BA | recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact generating organism | `0.95131` | `0.99042` | `0.89917` | `0.88810` | `0.81905` | `0.04286` |
| universal base Qwen3.5-27B control | `0.96065` | `0.99823` | `0.91056` | `0.92024` | `0.89524` | `0.05476` |
| universal Nemotron control | `0.95655` | `0.98708` | `0.91583` | `0.89762` | `0.82619` | `0.03095` |
| universal Gemma control | `0.92917` | `0.96344` | `0.88347` | `0.86548` | `0.85476` | `0.12381` |

Matched scores had 159 unique values and 662 ties. Exact-organism scoring took
146.7 seconds across 12 remote organism sessions. The universal Qwen control
took 75.2 seconds across four remote sessions and had 138 unique scores.

On the 16 Qwen LoRA units, applying the generating adapter instead of using the
same base Qwen checkpoint changed mean AUROC by `-0.00641`. Instructed units
were at ceiling (`-0.00016`), while the eight varied units lost `-0.01266` on
average. The largest paired drops were `a-mo-1` (`-0.04125`) and `a-mo-4`
(`-0.04250`). Three units improved, seven tied, and six regressed. A
dataset-unit bootstrap for the varied mean has a 95% interval
`[-0.02641, 0.00000]`, so this is evidence of concentrated adapter suppression,
not a precise universal penalty.

Do not use `lora is None` as a router. That would hard-code the known organism
structure rather than implement the intended general activation-based OOD
principle. Also note that AUROC is insensitive to cross-organism calibration in
this benchmark because every dataset unit has one homogeneous organism. If a
future hidden scoring unit mixes organisms, raw binary margins from different
model families may require a label-blind common calibration.
