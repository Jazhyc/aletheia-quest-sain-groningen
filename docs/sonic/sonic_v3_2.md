# sonic v3.2 — the judge ranks, the probe refines with an agreement-modulated cap

> Status: **submitted 2026-07-26 14:52. Official AUROC `0.903050` / BA
> `0.831250`, runtime 29m59s — the best official submission from this team on
> both metrics.** Supersedes `sonic_v3_1.ipynb` (AUROC `0.9021` / BA `0.8258`).
> Shipped with the inherited threshold `0.20`, which transferred. Per-unit
> results and the head-to-head against v3.1 and the pure judge:
> `docs/official_submissions.md`, "Sonic v3.2 counted detail".

## Problem

v3.1 used a fixed cap of 2 judge steps on every row.
This kept Notus safe.
The probe does not transfer to Notus.
The fixed cap prevented the probe from damaging the judge's ranking there.

But the same fixed cap limited the probe on Iris.
The probe is strong on Iris.
It scored AUROC `0.9950` in v2.3.5.
Under the fixed 2-step cap it only added `+0.017` over the judge alone.
The probe had more signal to give.

The two datasets differ in a measurable way.
On Iris the probe and the judge agree on direction.
Both flag deception on LoRA-adapted models.
On Notus the probe is near-random.
The product `judge_z × probe_z` is noise there.
A per-row signal can detect the difference.

## Rule

The cap is now per-row.
It is modulated by judge/probe agreement.

```
agreement = clip(judge_z × probe_z / AGREEMENT_SCALE, 0, 1)
cap = BASE_CAP + agreement × (MAX_CAP − BASE_CAP)
score = sigmoid(judge_z + cap × tanh(PROBE_GAIN × probe_z))
```

Three regimes from the product:

| product `judge_z × probe_z` | agreement | cap | behaviour |
|---|---|---|---|
| ≤ 0 (disagree) | 0 | BASE_CAP (2 steps) | same safety as v3.1 |
| ≥ AGREEMENT_SCALE (strongly agree) | 1 | MAX_CAP (4 steps) | double v3.1's probe influence |
| between 0 and AGREEMENT_SCALE | linear | interpolated | smooth transition |

On Iris deceptive rows the judge and the probe both point positive.
The product is large.
The cap opens toward MAX_CAP.
The probe adds more ranking signal.

On Iris honest rows both point negative.
The product is also large.
The cap opens too.
The probe correctly ranks honest rows there.
This does no harm.

On Notus the probe is random.
The product is noise.
The cap stays near BASE_CAP.
The safety guarantee holds.

`tanh` keeps the correction monotone in the probe.
No two rows are ever re-tied.

The threshold shipped at `0.20`, inherited from v3.1 rather than refitted.
It transferred: official BA is `0.8312`, above v3.1's `0.8258`.

## Constants

| constant | value | source |
| --- | --- | --- |
| JUDGE_SD | 1.200 | local judge cache |
| judge step z | 0.104 | JUDGE_ULP / JUDGE_SD |
| BASE_CAP | 2 steps = 0.208 | v3.1 fixed cap, validated safe on Notus |
| MAX_CAP | 4 steps = 0.417 | sweep (see below) |
| AGREEMENT_SCALE | 3.0 | sweep (see below) |
| PROBE_GAIN | 1.0 | fixed |
| PROBE_MEAN (qwen) | -0.252 | shipped probe on dev |
| PROBE_MEAN (gemma) | -1.968 | shipped probe on dev |
| PROBE_MEAN (nemotron) | -3.219 | shipped probe on dev |
| PROBE_SD (qwen) | 4.158 | shipped probe on dev |
| PROBE_SD (gemma) | 4.342 | shipped probe on dev |
| PROBE_SD (nemotron) | 2.194 | shipped probe on dev |

All constants are frozen offline.
None is derived from the scored batch.

## How MAX_CAP and AGREEMENT_SCALE were chosen

`fit_bounded_refine_v3_2.py` sweeps 15 combinations.
The candidates are MAX_CAP ∈ {4, 6, 8} steps.
The candidates are AGREEMENT_SCALE ∈ {1, 2, 3, 5, 8}.

Each combination is tested on 20 leave-one-organism-out dev folds.
These are the same folds v3.1 used.

The probe is degraded to 4 quality levels.
Independent Gaussian noise is added.
The AUROC targets are 0.70, 0.76, 0.85, and as-is.
The level 0.76 matches the known Notus probe quality.

The selection criterion is worst-fold mean Δ at probe quality 0.76.
This is the Notus regime.
Safety there is the binding constraint.

| MAX steps | AGREE SCALE | mean Δ @0.76 | worst fold Δ @0.76 | as-is Δ |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1.0 | +0.0036 | -0.0014 | +0.0085 |
| 4 | 3.0 | **+0.0036** | **-0.0012** | +0.0079 |
| 4 | 5.0 | +0.0036 | -0.0012 | +0.0078 |
| 6 | 3.0 | +0.0036 | -0.0013 | +0.0082 |
| 8 | 3.0 | +0.0035 | -0.0015 | +0.0085 |
| v3.1 fixed cap | — | +0.0036 | -0.0012 | +0.0076 |

The surface is flat.
Many combinations are nearly equal.
The selected combination matches v3.1 on Notus safety.
It gains slightly on the as-is regime.

The simulation cannot show the full Iris benefit.
The probe is in-sample on all dev folds.
The probe-judge agreement is inflated there.
The fixed cap already captures most of the available gain.
The real value of the agreement modulation is on out-of-sample Iris data.

If the official Iris gain is too small, MAX_CAP can be raised.
6 or 8 steps have a negligible Notus penalty.
The worst fold is -0.0013 to -0.0015.
This is within noise.

**Outcome.** The official Iris gain was `+0.0022` over v3.1 (`0.9405` -> `0.9427`),
concentrated on Iris/Qwen (`+0.0055`); Iris/gemma did not move at
all, so the cap never opened there.
Notus was flat as designed (`-0.0004` vs v3.1).

**MAX_CAP was raised, and it did not pay.** `sonic_v3_3` ran 6 steps on
2026-07-27 and Iris fell to `0.9393` (`-0.0034`), giving back more than the
`+0.0022` the modulation bought. That run also swapped the probe, so the two
changes are confounded and the cap is the suspect, not the proven cause. Either
way this sweep is the wrong instrument for the question: it scored candidates on
worst-fold Notus safety only, and the probe is in-sample on every fold, so the
Iris side of a larger cap was never priced. See `sonic_v3_3.md`.

## Comparison

| | v3 (convex blend) | v3.1 (fixed cap) | v3.2 (agreement cap) |
|---|---|---|---|
| rule | weighted sum of probe_z and judge_z | judge_z + fixed cap × tanh(probe_z) | judge_z + per-row cap × tanh(probe_z) |
| probe influence | 0.65 weight, unbounded | bounded at 2 steps | bounded at 2–4 steps, varies per row |
| Notus safety | failed (-0.1055) | flat (-0.0003) | flat (-0.0006) |
| Iris gain (official) | +0.0427 | +0.0169 | +0.0191 |
| headline AUROC (official) | 0.8624 | 0.9021 | **0.9031** |
| batch statistics | none | none | none |
| ties | none | none | none |

## Reproducing

```bash
# 1. sweep and fit constants
python experiments/ensemble_gate_eval/fit_bounded_refine_v3_2.py

# 2. build the notebook
python experiments/ensemble_gate_eval/build_sonic_v3_2_notebook.py
pytest experiments/ensemble_gate_eval/test_sonic_v3_2_notebook.py

# 3. benchmark on NDIF (repeat per dataset)
python experiments/ensemble_gate_eval/benchmark_ndif.py \
    --notebook submission/sonic_v3_2.ipynb --limit 400 \
    --output-dir results/sonic_v3_2/bench_varied_qwen_base

# 4. refit the threshold from NDIF runs, rebuild
python experiments/ensemble_gate_eval/fit_threshold_ndif.py \
    --constants results/ensemble_gate_eval/sonic_v3_2_constants.json --write
python experiments/ensemble_gate_eval/build_sonic_v3_2_notebook.py
```

Cells 2 and 4–9 are byte-identical to the v2.3.7 base.
Cell 11 (the judge) is identical to v3 and v3.1.
Only cell 12 (the score) changed.
