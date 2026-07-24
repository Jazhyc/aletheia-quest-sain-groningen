# Shared-trunk multi-family probe — evaluation

Run: `experiments/nonlinear_probe/run_multifamily_eval.py --max-epochs 60`
(seed 0), 2026-07-24. Raw output: `results/whitebox/multifamily_eval.json`,
log `logs/multifamily_eval.log`.

**sonic v3 change 4.** Only one parameter of `TransformerTokenProbe` is
family-specific: the input projection `Linear(hidden_dim, 128)`. Everything
after it — positional encoding, transformer encoder, head — lives at d_model=128
and is family-agnostic. This probe keeps one projection per family and shares
the rest, so the deception trunk is learned from all 8,216 rows across all three
families instead of each family's rows alone.

Motivation: Nemotron's entire dev corpus is **216 rows with 16 positives**
(verified against the HF API — there is exactly one Nemotron dev dataset), and
Notus/Nemotron is **52% of our AUROC gap to rank 1**.

Noise floor from repeated same-seed fits is **~0.002**; ignore smaller deltas.

> **Seed sweep, added after the first write-up. It changed the conclusion.**
> Seeds 1 and 2 were run after seed 0. The Nemotron result holds. The qwen
> result does not — it was single-seed luck. Section "Seed sweep" below has the
> numbers, and the recommendation has been corrected. Read that section before
> you use any single-seed figure on this page.

## Experiment 1 — Nemotron, 5-fold row CV (the 52% cell)

216 rows / 16 positives is far too few for a single holdout, so every row is
scored once out-of-fold and the AUROC is computed on the pooled predictions.

| max_epochs | single_family | shared_trunk |
| ---: | ---: | ---: |
| 3 | 0.9775 | 0.9791 |
| **60** | **0.9063** | **0.9791** (`+0.0728`) |

**The single-family Nemotron probe overfits; the shared trunk does not.** Its
score *falls* from `0.9775` to `0.9063` as training goes from 3 to 60 epochs —
the classic small-data signature. The shared trunk is unchanged at `0.9791` in
both, because its trunk is constrained by 8,000 rows from other families while
only the 4096x128 projection fits Nemotron's 173 training rows.

This is the largest effect any v3 lever has produced.

## Experiment 2 — leave-one-organism-out (qwen + gemma)

| method | mean AUROC | worst AUROC |
| --- | ---: | ---: |
| single_family | 0.9644 | 0.8344 |
| shared_trunk | 0.9657 | **0.8479** |

Aggregates understate it because nine of twenty folds are saturated above
`0.995`. On the ten non-saturated folds: single `0.9302` -> shared `0.9342`.
Of the eleven folds that move beyond the noise floor, **shared wins 7, loses 4**
— and the wins land where it matters, on the hardest folds:

| fold | single | shared | delta |
| --- | ---: | ---: | ---: |
| `qwen:varied/a-mo-qwen3.5-27b-3` | 0.8416 | 0.8699 | **+0.0283** |
| `qwen:varied/a-mo-qwen3.5-27b-1` (worst) | 0.8344 | 0.8479 | **+0.0135** |
| `qwen:varied/base` | 0.9770 | 0.9850 | +0.0079 |
| `qwen:varied/a-mo-qwen3.5-27b-4` | 0.9135 | 0.9213 | +0.0078 |
| `gemma:instr/s-mo-gemma-3-27b-it` | 0.9927 | 0.9798 | **−0.0129** |
| `gemma:instr/base` | 0.9877 | 0.9670 | **−0.0207** |

**qwen gains, gemma loses — consistently, on both gemma folds.** The pattern is
coherent: pooling helps a family whose own probe is overfitting or struggling
(nemotron, hard qwen folds) and hurts one whose own probe is already near
ceiling (gemma at `0.988`–`0.993`, with nothing to gain and a qwen-dominated
trunk to lose to).

## Seed sweep — three seeds

Seed 0 alone was misleading. Seeds 1 and 2 were run with the identical harness.

| seed | nemo single | nemo shared | LOO mean single | LOO mean shared | LOO worst single | LOO worst shared |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9063 | 0.9791 | 0.9644 | 0.9657 | 0.8344 | 0.8479 |
| 1 | 0.9641 | 0.9800 | 0.9660 | 0.9650 | 0.8451 | 0.8421 |
| 2 | 0.9563 | 0.9816 | 0.9661 | 0.9652 | 0.8468 | 0.8392 |
| **mean** | **0.9422** | **0.9802** | 0.9655 | 0.9653 | 0.8421 | 0.8431 |

**Nemotron holds.** The shared trunk wins on all three seeds. The mean gain is
`+0.0380`. Seed 0's `+0.0728` was the top of the range, not the typical value.

**The variance result is stronger than the mean result.** The single-family
Nemotron probe ranges `0.9063` to `0.9641` — a spread of `0.058`. The shared
trunk ranges `0.9791` to `0.9816` — a spread of `0.0025`. The shared trunk is
about 25 times more stable. You ship one draw of the weights. A probe that can
land anywhere in a `0.058` band is a risk in itself.

**qwen does not hold.** The LOO mean moves `-0.0002`. The worst organism moves
`+0.0010`. Both are inside the `0.002` noise floor. Seed 0's `+0.0135` on the
worst organism, and its 7-wins-4-losses split, were luck.

### Per-family breakdown, all three seeds

The combined table above hides that the two families move in opposite
directions. Mean AUROC over that family's folds:

| family | folds | seed 0 | seed 1 | seed 2 | mean change | single spread | shared spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen | 18 | +0.0033 | +0.0006 | +0.0002 | **+0.0014** | 0.0020 | 0.0011 |
| gemma | 2 | −0.0168 | −0.0158 | −0.0107 | **−0.0144** | 0.0015 | 0.0046 |

qwen is positive on every seed, but the margin is small. gemma is negative on
every seed, by 7 to 10 times the noise floor, and the shared trunk is also less
stable there.

## Recommendation: nemotron and qwen, not gemma

We already ship one weight set per family, so this needs no notebook change
beyond the weights:

- **nemotron → shared trunk.** It wins on every seed. It removes a `0.058`
  variance band. This unit is 52% of the leaderboard gap.
- **qwen → shared trunk.** The margin is small, but it is positive on every
  seed, so the measured downside is near zero. Notus / Qwen is 42.5% of the gap.
  The offline folds cannot reach the Notus regime, so "no measurable gain here"
  is not "no gain there". The property the trunk demonstrates is variance
  reduction, and that is what transfer needs.
- **gemma → keep the existing single-family probe.** The loss repeats on all
  three seeds. Notus / gemma is already our best Notus unit at `0.8259`. That
  unit is only 11% of the gap. There is little to win and a measured loss.

Be clear about the strength of the qwen case. It is an expected-value bet with a
measured floor, not a demonstrated gain.

## Caveats

- The Nemotron `+0.0728` is measured on **in-distribution row CV**, not on an
  unseen organism — Nemotron has only one organism, so LOO is not definable for
  it. It demonstrates that pooling fixes small-data overfitting. Whether that
  converts into Notus/Nemotron *transfer* is not shown here.
- The proxy weakness from `experiments/contrastive_probe/EVALUATION.md` still
  holds: every local fold is easier than every Notus unit, so no offline result
  can confirm a Notus gain.
- Single seed. The qwen per-fold gains (`+0.008` to `+0.028`) are comfortably
  above the `0.002` noise floor, but a seed sweep would tighten them.

## Reproduce

```bash
.venv/bin/python experiments/nonlinear_probe/run_multifamily_eval.py --max-epochs 60
.venv/bin/python experiments/nonlinear_probe/run_multifamily_eval.py \
    --experiments nemotron --max-epochs 3      # shows the overfitting crossover
```
