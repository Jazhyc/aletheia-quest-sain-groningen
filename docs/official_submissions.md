# Official Leaderboard Submissions

Durable record of official leaderboard submissions for team `SAIN Groningen`.
Use this for cross-method reporting across black-box prompts, Phoenix Wright
variants, text baselines, and future approaches.

Source: leaderboard `/api/me`, last queried on 2026-07-11. Official metrics are
the leaderboard's counted mean across private datasets. Dataset names are the
opaque public labels returned by the Space.

## Current Status

- Team: `SAIN Groningen`
- Pending submissions: `0`
- Rate-limit state at last query: `1` remaining in a `0.5` hour window
- Best official submission in this record: `phoenix_wright_v2_0.ipynb`
  compact privileged-distillation adapter, balanced accuracy `0.833333`

## Successful Submissions

| submitted at (UTC) | notebook | method / local mapping | tag | BA | AUROC | recall | FPR | runtime |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-07-11 09:47:58 | `phoenix_wright_v2_0.ipynb` | compact privileged-distillation adapter; mixed-data `5e-5` selection | black | 0.833333 | 0.833333 | 0.741667 | 0.075000 | 12476.5s |
| 2026-07-08 11:55:36 | `phoenix_wright_v1_3.ipynb` | context omission guard | black | 0.760833 | 0.760817 | 0.545000 | 0.023333 | 1928.9s |
| 2026-07-08 10:54:35 | `phoenix_wright_v1_3.ipynb` | details no-thinking prompt | black | 0.731667 | 0.732252 | 0.494167 | 0.030833 | 1910.5s |
| 2026-07-08 10:01:06 | `phoenix_wright_v1_3.ipynb` | truth-value guard | black | 0.775833 | 0.775779 | 0.608333 | 0.056667 | 1777.8s |
| 2026-07-07 21:36:59 | `phoenix_wright_v1_3.ipynb` | reasoning-output Phoenix, pre no-thinking prompt switch | black | 0.641250 | 0.641250 | 0.376667 | 0.094167 | 1955.0s |
| 2026-07-07 15:03:47 | `phoenix_wright_v1_3.ipynb` | reasoning-output Phoenix, earlier official run | black | 0.640833 | 0.640833 | 0.376667 | 0.095000 | 2120.5s |
| 2026-07-06 23:27:11 | `phoenix_wright_v1_3.ipynb` | degenerate Phoenix run; all-zero behavior | black | 0.500000 | 0.500000 | 0.000000 | 0.000000 | 4593.1s |
| 2026-07-05 02:08:07 | `bge_baseline.ipynb` | BGE small CPU text classifier | black | 0.508750 | 0.513121 | 0.385000 | 0.367500 | 561.0s |
| 2026-07-05 01:15:47 | `alminilm_baseline.ipynb` | MiniLM CPU text classifier | black | 0.505417 | 0.501275 | 0.446667 | 0.435833 | 564.4s |
| 2026-07-04 22:55:14 | `ngrams_baseline.ipynb` | cached text n-gram probe baseline | black | 0.512500 | 0.523558 | 0.352500 | 0.327500 | 170.0s |
| 2026-07-04 22:22:11 | `random_baseline.ipynb` | random/no-NDIF sandbox health baseline | untagged | 0.475000 | 0.476121 | 0.455000 | 0.505000 | 124.2s |

## Recent Phoenix Counted Dataset Detail

These counted Notus/Iris rows are useful diagnostics for official transfer.

### Phoenix Wright v2 Compact Distillation

Submitted 2026-07-11 09:47:58 UTC. Headline BA `0.833333`.

| dataset | model / lora | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dataset Notus | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `None` | 0.7475 | 0.7475 | 0.5800 | 0.0850 |
| Dataset Notus | `Qwen/Qwen3.5-27B` / `None` | 0.7075 | 0.7075 | 0.5150 | 0.1000 |
| Dataset Notus | `google/gemma-3-27b-it` / `None` | 0.7450 | 0.7450 | 0.6200 | 0.1300 |
| Dataset Iris | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `g-st-nemotron-3-super-120b` | 0.9275 | 0.9275 | 0.9100 | 0.0550 |
| Dataset Iris | `Qwen/Qwen3.5-27B` / `g-st-qwen3.5-27b` | 0.9375 | 0.9375 | 0.9200 | 0.0450 |
| Dataset Iris | `google/gemma-3-27b-it` / `g-st-gemma-3-27b-it-2` | 0.9350 | 0.9350 | 0.9050 | 0.0350 |

Notes: this is the first successful compact-distillation submission and the
best official result so far. Relative to the best v1.3 prompt judge, it raises
headline BA by 0.0575 and recall by 0.1333, at the cost of FPR increasing by
0.0183. Iris transfer is particularly strong. The submitted adapter used the
earlier mixed-data training selection; the later varied-only AdamW adapter was
slightly stronger on local test but was not included in this run. Runtime was
about 3h28m, much longer than the roughly 32-minute v1.3 runs.

### Context Omission Guard

Submitted 2026-07-08 11:55:36 UTC. Headline BA `0.760833`.

| dataset | model / lora | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dataset Notus | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `None` | 0.6975 | 0.6975 | 0.4150 | 0.0200 |
| Dataset Notus | `Qwen/Qwen3.5-27B` / `None` | 0.6325 | 0.6324 | 0.3050 | 0.0400 |
| Dataset Notus | `google/gemma-3-27b-it` / `None` | 0.7350 | 0.7350 | 0.4900 | 0.0200 |
| Dataset Iris | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `g-st-nemotron-3-super-120b` | 0.8450 | 0.8450 | 0.7100 | 0.0200 |
| Dataset Iris | `Qwen/Qwen3.5-27B` / `g-st-qwen3.5-27b` | 0.8275 | 0.8275 | 0.6700 | 0.0150 |
| Dataset Iris | `google/gemma-3-27b-it` / `g-st-gemma-3-27b-it-2` | 0.8275 | 0.8275 | 0.6800 | 0.0250 |

Notes: lower FPR than truth-value guard, but recall dropped enough that headline
BA was lower. It improved over details no-thinking, but did not beat the
truth-value guard.

### Details No-Thinking Prompt

Submitted 2026-07-08 10:54:35 UTC. Headline BA `0.731667`.

| dataset | model / lora | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dataset Notus | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `None` | 0.7125 | 0.7117 | 0.4550 | 0.0300 |
| Dataset Notus | `Qwen/Qwen3.5-27B` / `None` | 0.6700 | 0.6692 | 0.4050 | 0.0650 |
| Dataset Notus | `google/gemma-3-27b-it` / `None` | 0.7025 | 0.7076 | 0.4500 | 0.0450 |
| Dataset Iris | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `g-st-nemotron-3-super-120b` | 0.7800 | 0.7800 | 0.5700 | 0.0100 |
| Dataset Iris | `Qwen/Qwen3.5-27B` / `g-st-qwen3.5-27b` | 0.7600 | 0.7600 | 0.5250 | 0.0050 |
| Dataset Iris | `google/gemma-3-27b-it` / `g-st-gemma-3-27b-it-2` | 0.7650 | 0.7650 | 0.5600 | 0.0300 |

Notes: Notus was close to truth-value guard, but Iris dropped sharply due to low
recall. This prompt was too conservative officially.

### Truth-Value Guard

Submitted 2026-07-08 10:01:06 UTC. Headline BA `0.775833`.

| dataset | model / lora | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dataset Notus | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `None` | 0.7100 | 0.7079 | 0.4950 | 0.0750 |
| Dataset Notus | `Qwen/Qwen3.5-27B` / `None` | 0.6625 | 0.6624 | 0.3750 | 0.0500 |
| Dataset Notus | `google/gemma-3-27b-it` / `None` | 0.7100 | 0.7118 | 0.4800 | 0.0600 |
| Dataset Iris | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `g-st-nemotron-3-super-120b` | 0.8650 | 0.8650 | 0.7700 | 0.0400 |
| Dataset Iris | `Qwen/Qwen3.5-27B` / `g-st-qwen3.5-27b` | 0.8650 | 0.8650 | 0.7750 | 0.0450 |
| Dataset Iris | `google/gemma-3-27b-it` / `g-st-gemma-3-27b-it-2` | 0.8425 | 0.8425 | 0.7550 | 0.0700 |

Notes: best official Phoenix run so far. Its advantage over context omission
comes from much stronger Iris recall, despite higher FPR.

## Failed Official Attempts

The API history includes multiple failed Phoenix Wright submissions before the
remote NDIF session shape and packaging were hardened.

| date range (UTC) | notebooks | failed dataset | rough pattern |
| --- | --- | --- | --- |
| 2026-07-03 to 2026-07-04 | `phoenix_wright_v0.ipynb`, `phoenix_wright_v1.ipynb`, `phoenix_wright_v1_1.ipynb`, `phoenix_wright_v1_2.ipynb` | Dataset Metis | sandbox failures during early Phoenix Wright submission bring-up |
| 2026-07-06 to 2026-07-07 | `phoenix_wright_v1_3.ipynb` | Dataset Metis / Dataset Eunomia | remote NDIF/session reliability failures before one-session `VisionLanguageModel` hardening and batch-size stabilization |

For current official status, query the Space directly with `/api/me` rather than
inferring from this file. After each completed official submission, add a row
above and record any counted Notus/Iris diagnostics that materially change
method selection.
