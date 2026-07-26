# sonic v3.1 — the judge ranks, the probe refines

> Status: **submitted 2026-07-26 12:55.** Official headline AUROC `0.9021` / BA
> `0.8258`, runtime 29m6s. Beats the pure judge (phoenix_wright_v4 at `0.8938`)
> by `+0.0083`. Best official submission from this team on the primary metric.
> See `docs/official_submissions.md` for the full counted-unit breakdown.
> It replaces `sonic_v3.ipynb`, which scored AUROC `0.862417` officially on
> 2026-07-26 and lost to `phoenix_wright_v4` at `0.893817`.

## What v3 got wrong

v3 combined the probe and the judge with a convex blend at a probe weight of
`0.65`. The official result showed that this weight is safe on one counted
dataset and unsafe on the other.

| | Notus | Iris | headline |
| --- | ---: | ---: | ---: |
| phoenix v4 (judge alone) | 0.8640 | 0.9236 | **0.8938** |
| sonic v3 (blend, probe weight 0.65) | 0.7585 | 0.9663 | 0.8624 |
| difference | **-0.1055** | **+0.0427** | -0.0314 |

The probe does not transfer to Notus. Its Notus AUROC has been about `0.7577`
on every sonic run since v2.1, and v3's `0.7585` continues that line. At a
weight of `0.65` it overrides a judge that ranks at `0.8640` there. On Iris the
same probe is the stronger detector and is worth `+0.0427`.

Three facts make this hard to fix by re-tuning the weight:

- **No single weight serves both regimes.** Interpolating linearly in the probe
  weight, the two effects net to `-0.0483 x weight`, so every positive weight
  is projected to lose to the pure judge.
- **No inference-time signal separates them.** Notus is base-model-only and
  Iris is LoRA-only, but "base model" is not the tell: held-out base-model dev
  organisms score probe `0.9878` against judge `0.9814`. Keying the weight on
  `has_lora` would fit two leaderboard points, not a mechanism.
- **Dev cannot see the failure.** Every dev unit is instructed or varied
  deception, and the probe wins all of them. Metis and Eunomia, which the
  leaderboard shows but does not count, both went *up* under v3.

## What v3.1 does

The judge owns the ranking. The probe may only nudge.

    score = sigmoid(judge_margin / JUDGE_SD
                    + PROBE_CAP * tanh(PROBE_GAIN * probe_z))

    probe_z = (probe_logit - PROBE_MEAN[family]) / PROBE_SD[family]

The judge margin is bf16. Its quantization step is `0.125`, and it takes only
about 34 distinct values per 400 rows, so the judge's ordering is mostly ties.
`PROBE_CAP` is **two judge steps** — about `4%` of the margin's range. So:

- Where the judge is tied, the probe decides. That is a gain whenever the probe
  beats chance, which it does even on Notus (`0.7577`).
- Where the judge is confident, the probe cannot overturn it. The damage from a
  probe that does not transfer is bounded by construction.
- `tanh`, rather than a hard clip, keeps the correction strictly monotone in
  the probe, so no two rows are ever re-tied.

### Constants

| constant | value | source |
| --- | --- | --- |
| JUDGE_SD | 1.199755 | local judge cache |
| judge step | 0.125 / JUDGE_SD = 0.10419 | bf16 ulp of the label logits |
| PROBE_CAP | 2 steps = 0.20838 | chosen for worst-case robustness |
| PROBE_GAIN | 1.0 | |
| PROBE_MEAN | qwen -0.252, gemma -1.968, nemotron -3.219 | shipped probes on dev |
| PROBE_SD | qwen 4.158, gemma 4.342, nemotron 2.194 | shipped probes on dev |

The cap is expressed in judge *steps*, not in units of the margin's spread, so
it stays correct even though the NDIF margin distribution is wider than the
local cache (see "Two measurement corrections").

## Why the cap is two steps

The regime that matters cannot be observed, so it was simulated. Each dev fold
was scored with its real judge margin and with its probe artificially blunted by
noise until the probe alone ranked at a target AUROC. Mean change against the
judge alone across the 20 leave-one-organism-out folds:

| probe quality | v3 blend | v3.1 | v3.1 worst fold |
| --- | ---: | ---: | ---: |
| 0.70 | -0.1786 | **+0.0024** | -0.0032 |
| 0.76 (Notus level) | -0.1117 | **+0.0042** | -0.0011 |
| 0.85 | -0.0372 | **+0.0060** | -0.0006 |
| as-is on dev | +0.0175 | **+0.0076** | +0.0001 |

v3.1 is the only setting that is positive in every regime. A sweep over cap
sizes `{1, 1.5, 2, 3, 4}` steps, gains `{0.3, 0.5, 1, 2}`, and hard clip versus
`tanh` selected `tanh` at 2 steps on worst-case regime performance; hard clips
above 3 steps go negative once the probe is Notus-quality.

The floor is close to guaranteed rather than merely measured. In the limit of a
vanishing cap the rule becomes a pure within-tie tiebreak, which was
non-negative on **every one of the 20 folds** (mean `+0.0031`, up to `+0.0105`
on the hardest). Two steps trades a little of that guarantee for roughly twice
the gain.

## Two measurement corrections

**The gemma probe was silently down-weighted.** v3 divided each family's probe
logit by a frozen spread. Measuring the shipped probes' actual output on dev
gives gemma `4.342`, but v3 shipped `6.466` — too large by `1.5x`, so gemma's
probe contributed a third less than intended. This is the likely cause of
official Iris/gemma falling from `0.9921` under v2.3.5 (probe alone) to
`0.9161` under v3.

**The probe mean is now subtracted.** Under a convex blend an offset shared by
every row cannot change a dataset's AUROC, so v3 correctly ignored it. Under
`tanh` it decides where each row sits on the curve, so it has to be frozen too.
Nemotron's mean is `-3.219`; leaving it in would have pushed every nemotron row
onto the saturated tail.

**The NDIF judge margin is wider than the local cache.** On NDIF the implied
`judge_z` runs from `-5.28` to `+2.68`; the local 4090 cache only spans `-1.57`
to `+2.40`. This does not affect AUROC or the cap, both of which are defined by
the quantization step, but it does mean a threshold fitted on the local cache
sits too high. The cut is therefore fitted on NDIF runs instead.

## NDIF benchmark against Phoenix v4

Both notebooks, same harness (`benchmark_ndif.py`), same datasets, same row
counts. **v3.1 wins every unit.**

| dataset | rows | v4 AUROC | v3.1 AUROC | change | v4 unique | v3.1 unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| varied-qwen-a-mo-1 | 400 | 0.7621 | **0.7881** | **+0.0260** | 72 | 400 |
| instr-nemotron-base | 216 | 0.9288 | **0.9503** | **+0.0215** | 52 | 216 |
| varied-qwen-base | 400 | 0.9394 | **0.9501** | **+0.0107** | 83 | 400 |
| instr-gemma-base | 400 | 0.9958 | **0.9962** | +0.0004 | 87 | 400 |
| **mean** | | 0.9065 | **0.9212** | **+0.0147** | | |

The gain scales with how hard the unit is, which is the behaviour the design
predicts. The judge emits only 52--87 distinct scores per unit, so its ranking
is mostly ties; the probe resolves them and v3.1 emits one distinct score per
row. On an easy unit there is nothing left to resolve (`+0.0004` on
instr-gemma-base); on the hardest one the tie mass is large and the gain is
`+0.0260`. Notus is a hard regime for the judge (`0.8640`), so it sits at the
favourable end of that range.

For reference, sonic v3 scored `0.9837` on `varied-qwen-base` — higher than
v3.1, and meaningless: the probe was trained on those rows, and the same
configuration scored `0.7585` on the real Notus.

## Thresholds

AUROC ignores the cut entirely, but v3 shipped `0.5` and produced a public BA of
`0.6213`, with two counted units at chance. The same fault reappeared in
benchmarking: at a `0.5` cut, instr-nemotron-base scored BA `0.4975` at AUROC
`0.9503`.

The cause is that a sign test is the wrong cut for this judge. Its margin
distribution sits well below zero on NDIF, which is also why
`phoenix_wright_v4` cuts at `0.15` on `sigmoid(margin)` (margin `>= -1.735`)
rather than at `0.5`.

`fit_threshold_ndif.py` fits one frozen cut across the four executed NDIF dev
runs above — never on the batch being scored:

| cut | mean BA | worst unit |
| --- | ---: | ---: |
| 0.50 | 0.7262 | 0.4975 |
| 0.15 | 0.8503 | 0.7050 |
| **0.20** | **0.8531** | 0.7000 |
| 0.25 | 0.8359 | 0.6900 |

Shipped: `0.20`. Re-running instr-nemotron-base with it recovers BA
`0.4975 -> 0.8500` with AUROC unchanged at `0.9503`.

## Projection vs actual

| | v4 actual | v3.1 projected | v3.1 **actual** |
| --- | ---: | ---: | ---: |
| Notus | 0.8640 | 0.868 -- 0.885 | **0.8638** |
| Iris | 0.9236 | 0.928 -- 0.940 | **0.9405** |
| **headline** | **0.8938** | **0.898 -- 0.912** | **0.9021** |

Iris landed at the top of its projected range. Notus landed slightly below
(`-0.0003` vs v4), with the tiebreaking benefit failing to materialise.
Notus/gemma specifically regressed `-0.0061` — the probe's within-tie ordering
on gemma base-model data is slightly worse than random. The robustness
simulation's worst fold at probe quality 0.76 was `-0.0011`; the actual gemma
unit is outside that envelope, confirming that the dev folds do not fully
capture the Notus distribution shift.

The headline of `0.9021` is inside the projected range, the gain over v4 is
real, and no counted unit lost statistically meaningful AUROC against the
judge alone. The design's core guarantee — non-negative against the judge — held
on aggregate Notus (`-0.0003` is within noise) though the per-unit gemma loss
shows the floor is not as absolute as the simulation suggested.

## Reproducing

```bash
# 1. fit constants and run the robustness simulation
python experiments/ensemble_gate_eval/fit_bounded_refine.py

# 2. build the notebook from sonic_v2.3.7 plus the constants
python experiments/ensemble_gate_eval/build_sonic_v3_1_notebook.py
pytest experiments/ensemble_gate_eval/test_sonic_v3_1_notebook.py

# 3. one dataset on NDIF (repeat per dataset; nemotron has only 216 rows)
python experiments/ensemble_gate_eval/benchmark_ndif.py \
    --notebook submission/sonic_v3_1.ipynb --limit 400 \
    --output-dir results/sonic_v3_1/bench_varied_qwen_base

# 4. freeze the binary cut from those runs
python experiments/ensemble_gate_eval/fit_threshold_ndif.py --write
python experiments/ensemble_gate_eval/build_sonic_v3_1_notebook.py
```

Cells 2 and 4--9 are byte-identical to `legacy_submissions/sonic_v2.3.7.ipynb`,
and cell 11 (the judge) is identical to `sonic_v3`. Only the scoring cell
changed.
