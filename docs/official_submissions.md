# Official Leaderboard Submissions

Durable record of official leaderboard submissions for team `SAIN Groningen`.
Use this for cross-method reporting across black-box prompts, Phoenix Wright
variants, text baselines, and future approaches.

Source: leaderboard API, last queried on 2026-07-16. Official metrics are
the leaderboard's counted mean across private datasets. Dataset names are the
opaque public labels returned by the Space.

## Current Status

- Team: `SAIN Groningen`
- Pending submissions: `1` running (as of 2026-07-23)
- Rate-limit state at last query: `1` remaining in a `0.5` hour window
- Best official submission in this record: `sonic_v2.1.ipynb`
  white-box probe + judge gate, balanced accuracy `0.847083`, AUROC `0.869833`
- Leaderboard rank: `2` of the public board; leader is EleutherAI
  `Tiresias.ipynb` at AUROC `0.9451` / BA `0.8904` (see benchmark below)
- Previous best black-box entry: `phoenix_wright_v2_0.ipynb`, BA `0.833333`

The sonic family is white-box (`--tag white`). Unlike the Phoenix rows, its BA
and AUROC differ because the probe emits continuous scores. Sonic timestamps
below are the leaderboard-displayed times, not re-derived UTC.

The detailed API table below was last refreshed on 2026-07-16. On 2026-07-23,
Phoenix v3.1 and the reasoning-n-gram Phoenix v3.2 submission both returned the
same user-confirmed headline score, `0.8316667`. Because v3.2 did not transfer
its local projected gain, the active repository submission was reverted to
Phoenix v3.0 with the original rank-16 adapter. Per-dataset details for these
recent runs have not been added without a fresh authenticated API query.

## Successful Submissions

| submitted at (UTC) | notebook | method / local mapping | tag | BA | AUROC | recall | FPR | runtime |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-07-23 21:46 | `sonic_v2.2.ipynb` | constant-free gate: rank-space blend, `w = clip(2a-1,0,1)`, degeneracy guard, prevalence-matched threshold | white | 0.810417 | 0.858450 | 0.744167 | 0.123333 | 4705s |
| 2026-07-23 18:39 | `sonic_v2.1.ipynb` | soft blend `w = clamp((a-0.5)x3,0,1)` + threshold search against judge labels (**best official**) | white | 0.847083 | 0.869833 | 0.764167 | 0.070000 | 5398s |
| 2026-07-23 13:33 | `sonic_v2.ipynb` | first probe+judge gate; hard switch at agreement `0.65`, unsupervised GMM threshold | white | 0.747083 | 0.844700 | 0.580000 | 0.085833 | 4345s |
| 2026-07-20 01:14 | `sonic_v1.ipynb` | white-box transformer token probe at L46, probe alone (no judge) | white | 0.655000 | 0.754850 | 0.440833 | 0.130833 | 1046s |
| 2026-07-16 15:57:06 | `phoenix_wright_v2_4.ipynb` | same adapter; conditional 4,000-character reasoning, summary member only | black | 0.776250 | 0.776250 | 0.739167 | 0.186667 | 5131.7s |
| 2026-07-16 07:36:16 | `phoenix_wright_v2_3.ipynb` | same adapter; conditional 4,000-character reasoning plus summary/binary max ensemble | black | 0.729167 | 0.729167 | 0.857500 | 0.399167 | 6077.6s |
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

### Sonic family counted Notus/Iris detail

All four sonic runs share cells 1--11 (identical probe and judge); only the
gate in cell 12 changes. Counted balanced accuracy per unit:

| dataset / model | v1 | v2 | v2.1 | v2.2 |
| --- | ---: | ---: | ---: | ---: |
| Notus `Nemotron-120B` / `None` | 0.5025 | 0.7525 | 0.7525 | **0.5025** |
| Notus `Qwen3.5-27B` / `None` | 0.5700 | 0.5700 | 0.6950 | 0.6950 |
| Notus `gemma-3-27b-it` / `None` | 0.5000 | 0.7575 | 0.7575 | 0.7575 |
| Iris `Nemotron-120B` / `g-st` | 0.8725 | 0.9500 | 0.9450 | 0.9525 |
| Iris `Qwen3.5-27B` / `g-st` | 0.9850 | 0.9425 | 0.9500 | 0.9850 |
| Iris `gemma-3-27b-it` / `g-st-2` | 0.5000 | 0.5100 | 0.9825 | 0.9700 |
| **headline BA** | 0.6550 | 0.7471 | **0.8471** | 0.8104 |
| **headline AUROC** | 0.7549 | 0.8447 | **0.8698** | 0.8584 |

Counted AUROC per unit for the two newest runs:

| dataset / model | v2.1 AUROC | v2.2 AUROC |
| --- | ---: | ---: |
| Notus `Nemotron-120B` / `None` | 0.6900 | 0.5834 |
| Notus `Qwen3.5-27B` / `None` | 0.7572 | 0.7572 |
| Notus `gemma-3-27b-it` / `None` | 0.8259 | 0.8259 |
| Iris `Nemotron-120B` / `g-st` | 0.9905 | 0.9926 |
| Iris `Qwen3.5-27B` / `g-st` | 0.9643 | 0.9995 |
| Iris `gemma-3-27b-it` / `g-st-2` | 0.9911 | 0.9921 |

Notes: v2.2's entire regression against v2.1 is **one cell**, Notus/Nemotron,
where BA fell `0.7525 -> 0.5025` and AUROC `0.6900 -> 0.5834`. Those are exactly
`sonic_v1`'s probe-only values for that unit, so on Notus/Nemotron v2.2 scored
the bare probe with the default config threshold.

The cause is **not** the degeneracy guard. That guard tests
`np.std(...) < 1e-6` and sits inside the `judge_labels.any() and not
judge_labels.all()` branch, so both judge classes are already guaranteed
present; the binary verdicts then have std `>= ~0.5/sqrt(n)` and the soft
probabilities vary as well. `judge_degenerate` is therefore unreachable in
practice, and `probe_degenerate` needs a fully saturated sigmoid. The guard
never fires.

The bare-probe output comes from one of the two `else` branches that discard
the judge outright:

- `gate: no judge signal for this dataset` -- `judge_scores is None` (judge
  errored or was skipped) or a length mismatch.
- `gate: judge verdicts are single-class here` -- the judge returned all-0 or
  all-1 hard verdicts.

Both print a distinguishing line, so the captured notebook output of the
2026-07-23 21:46 run identifies which one fired. Runtime fell from 89m58s
(v2.1) to 78m25s (v2.2) with cells 1--11 unchanged, which mildly favours an
aborted judge pass, but a single-class verdict fits the metrics equally well.

The single-class branch is itself a latent bug: it throws away the judge's
**soft** probabilities merely because the thresholded verdicts collapsed to one
class. Soft confidences can rank well even when every hard verdict is 0, so
that path discards usable ranking signal.

Everywhere else v2.2 matched or beat v2.1: Notus/Qwen and Notus/gemma are
identical, and Iris improved on Nemotron (`+0.0075`) and Qwen (`+0.0350`), with
a small Iris/gemma cost (`-0.0125`). Iris AUROC under v2.2 is the best of any
run (mean `0.9947`). So the rank-space blend plus prevalence-matched threshold
is a genuine improvement; only the detector-dropping guard is harmful. A v2.3
that keeps v2.2's gate but never drops the judge projects to roughly BA `0.852`
/ AUROC `0.876`, above v2.1 on both.

Also note v1's inverted units: Notus/gemma AUROC `0.3526` (below chance, probe
ranks backwards) and Iris/gemma BA `0.5000` at AUROC `0.9911` (near-perfect
ranking, useless threshold). The v2.1 threshold search fixed the latter; v2.2's
agreement-sign flip covers the former.

### Benchmark: EleutherAI `Tiresias.ipynb` (rank 1)

Submitted 2026-07-20 04:18. Headline AUROC `0.9451` / BA `0.8904`, runtime
191m43s.

| dataset | model / lora | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dataset Notus | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `None` | 0.8500 | 0.9056 | 0.8500 | 0.1500 |
| Dataset Notus | `Qwen/Qwen3.5-27B` / `None` | 0.8550 | 0.9326 | 0.8550 | 0.1450 |
| Dataset Notus | `google/gemma-3-27b-it` / `None` | 0.8000 | 0.8715 | 0.8000 | 0.2000 |
| Dataset Iris | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `g-st-nemotron-3-super-120b` | 0.9400 | 0.9863 | 0.9600 | 0.0800 |
| Dataset Iris | `Qwen/Qwen3.5-27B` / `g-st-qwen3.5-27b` | 0.9650 | 0.9919 | 0.9700 | 0.0400 |
| Dataset Iris | `google/gemma-3-27b-it` / `g-st-gemma-3-27b-it-2` | 0.9325 | 0.9825 | 0.9300 | 0.0650 |

Notes: the gap to rank 1 is **entirely Notus**. On Iris we are already ahead --
v2.2 mean BA `0.9692` vs Tiresias `0.9458`, mean AUROC `0.9947` vs `0.9869`. On
Notus Tiresias averages BA `0.8350` / AUROC `0.9032` against our best `0.7350` /
`0.7577` (v2.1). Their Notus recall equals their BA with FPR = 1 - BA on every
unit, i.e. a well-calibrated balanced split, whereas ours under-flags (recall
`0.485--0.610` at FPR `0.085--0.095`). But our Notus deficit is a *ranking*
deficit, not a threshold one: at AUROC `0.7572`/`0.8259` no threshold reaches
their BA. Notus is base-model-only (`LoRA = None`) for all three families, the
regime where our probe transfers worst.

### Phoenix Wright v2.4 trace-summary isolation

Submitted 2026-07-16 15:57:06 UTC. Headline BA `0.776250`.

| dataset | model / lora | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dataset Notus | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `None` | 0.6800 | 0.6800 | 0.5100 | 0.1500 |
| Dataset Notus | `Qwen/Qwen3.5-27B` / `None` | 0.6075 | 0.6075 | 0.4850 | 0.2700 |
| Dataset Notus | `google/gemma-3-27b-it` / `None` | 0.6575 | 0.6575 | 0.6100 | 0.2950 |
| Dataset Iris | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `g-st-nemotron-3-super-120b` | 0.9150 | 0.9150 | 0.9650 | 0.1350 |
| Dataset Iris | `Qwen/Qwen3.5-27B` / `g-st-qwen3.5-27b` | 0.9150 | 0.9150 | 0.9350 | 0.1050 |
| Dataset Iris | `google/gemma-3-27b-it` / `g-st-gemma-3-27b-it-2` | 0.8825 | 0.8825 | 0.9300 | 0.1650 |

Notes: removing v2.3's binary recall member recovered `0.0471` BA and reduced
FPR by `0.2125`, confirming that the max ensemble caused most of that run's
extreme over-flagging. The trace-summary member still remained `0.0571` BA
below v2.0: recall was essentially unchanged (`0.7392` versus `0.7417`), while
FPR rose from `0.0750` to `0.1867`. FPR increased in every counted family.
This rules out a purely non-Qwen-family failure; counted Qwen also regressed,
especially on Notus, while many uncounted Metis/Eunomia Qwen units remained
strong. Treat this as response-mode/interface transfer failure, with
cross-family trace OOD and trace unfaithfulness both plausible contributors.

The backend is only a formal official-run confound, not a credible explanation
for this large systematic shift. The exact `LanguageModel` deployment path
matched the shared-vLLM predictions on all 40 labeled rows in its remote
rehearsal, and changing from one to two requests per session preserved 40/40
agreement. The more important implementation confound is prompt serialization:
v2.4 merged adjacent same-role messages, stopped applying the v2.1 1,500-character
per-message cap before the 3,000-character context cap, removed the context
truncation marker, and changed long-output head/tail allocation. Thus rows with
no reasoning received the ordinary judge instruction but not necessarily the
exact v2.1 rendered input. The official run does not isolate reasoning
visibility cleanly from these formatting changes in principle. A post-result
exact-render audit found that these differences changed only 7/6,573 local
training rows and changed 0/822 validation and 0/821 test rows. Moreover, the
historical no-trace and 0.9262 trace-visible local-test evaluations both used
the same newer formatter. Formatting therefore cannot explain the local gain
and is unlikely to explain the official regression unless the private counted
rows have a material long-message or turn-structure shift. The primary working
explanation remains reasoning-interface transfer from Qwen-only varied traces.

### Phoenix Wright v2.3 reasoning-trace ensemble

Submitted 2026-07-16 07:36:16 UTC. Headline BA `0.729167`.

| dataset | model / lora | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dataset Notus | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `None` | 0.6475 | 0.6475 | 0.6650 | 0.3700 |
| Dataset Notus | `Qwen/Qwen3.5-27B` / `None` | 0.5800 | 0.5800 | 0.7450 | 0.5850 |
| Dataset Notus | `google/gemma-3-27b-it` / `None` | 0.6125 | 0.6125 | 0.8450 | 0.6200 |
| Dataset Iris | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` / `g-st-nemotron-3-super-120b` | 0.8500 | 0.8500 | 0.9700 | 0.2700 |
| Dataset Iris | `Qwen/Qwen3.5-27B` / `g-st-qwen3.5-27b` | 0.8475 | 0.8475 | 0.9700 | 0.2750 |
| Dataset Iris | `google/gemma-3-27b-it` / `g-st-gemma-3-27b-it-2` | 0.8375 | 0.8375 | 0.9500 | 0.2750 |

Notes: reject this interface. Relative to v2.0--v2.2, recall rose from
0.7417 to 0.8575 but FPR rose from 0.0750 to 0.3992, reducing BA by 0.1042.
The same weights were used, so the regression is attributable to the changed
inference interface rather than a new adapter. The trace-visible and binary-max
components cannot be separated from this aggregate official run, but both were
selected on Qwen-heavy local splits and are unsafe under the counted response-
mode shift. The submission notebook was rolled back to the v2.1 no-trace,
single-summary decision rule while retaining `LanguageModel` and mechanical
batching optimizations. Do not restore reasoning traces, the binary max member,
or the logical regex override without a new cross-mode validation design.

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
| 2026-07-16 | `phoenix_wright_v2_4.ipynb` | Dataset Eunomia | per-dataset timeout after 2559.0s total: Metis used about 759s, then Eunomia exhausted the 1800s sandbox budget; consistent with an NDIF/session stall rather than an import or packaging failure |
| 2026-07-03 to 2026-07-04 | `phoenix_wright_v0.ipynb`, `phoenix_wright_v1.ipynb`, `phoenix_wright_v1_1.ipynb`, `phoenix_wright_v1_2.ipynb` | Dataset Metis | sandbox failures during early Phoenix Wright submission bring-up |
| 2026-07-06 to 2026-07-07 | `phoenix_wright_v1_3.ipynb` | Dataset Metis / Dataset Eunomia | remote NDIF/session reliability failures before one-session `VisionLanguageModel` hardening and batch-size stabilization |

For current official status, query the Space directly with `/api/me` rather than
inferring from this file. After each completed official submission, add a row
above and record any counted Notus/Iris diagnostics that materially change
method selection.
