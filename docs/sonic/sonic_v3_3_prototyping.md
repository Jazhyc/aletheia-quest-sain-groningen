# sonic v3.3 prototyping — probe experiments

> Status: experiments only. **Nothing here has shipped.** `submission/` still
> holds `sonic_v3_2.ipynb` with the `whitebox_probe_v2_3_7` weights (official
> AUROC `0.903050` / BA `0.831250`).

Scope: v3.3 changes the **probe**, not the scoring rule. v3.2's cell 12 stays
byte-identical; only the probe weights and the `PROBE_LOGIT_MEAN`/`SD`
constants would move. Same shape as the v2.3.7 weights-only swap.

A gate design keyed on the LoRA field was built and **rejected** on 2026-07-26
without submission — switching scoring formulas per regime reads as
"fundamentally different detection *logic* per organism", which the competition
rules prohibit. See `docs/sonic/README.md`.

Background, data inventory and the external-data search:
`docs/probe_training_and_external_data.md`.

## The harness

`experiments/nonlinear_probe/train_v3_3_probe.py` — leave-one-cell-out over the
four `(scenario x adapter)` cells. For each held-out cell the probe trains on
every row outside it and scores the cell it never saw.

| cell | rows | families present |
| --- | ---: | --- |
| instr/adapter | 3600 | qwen, gemma |
| instr/base | 1016 | qwen, gemma, nemotron |
| varied/adapter | 3200 | qwen |
| varied/base | 400 | qwen |

**Noise floor: ~0.0012.** Re-running an identical configuration (same seed, same
data, same code) gave `varied/base` = 0.9867 on one run and 0.9879 on another,
from GPU non-determinism. Treat single-cell gaps below ~0.002 as ties, and
prefer sign-consistency across cells over the mean.

## Experiment 1 — training recipe

Does fixing the early-stopping split help? The shipped `MultiFamilyProbe.fit`
selected checkpoints with `train_test_split(..., stratify=labels)` — a random
row split, so validation rows came from the same organisms and scenarios as
training rows and early stopping rewarded memorisation, not transfer.

- `random` — the shipped recipe
- `grouped` — early stopping holds out whole organisms
- `balanced` — grouped, plus base rows upweighted from 17% to ~50% of the loss

| held-out cell | random | grouped | balanced |
| --- | ---: | ---: | ---: |
| instr/adapter | 0.9934 | 0.9949 | **0.9950** |
| instr/base | 0.9902 | 0.9799 | **0.9911** |
| varied/adapter | 0.8922 | 0.9030 | **0.9034** |
| varied/base | 0.9851 | 0.9839 | **0.9867** |

**Verdict: adopt `balanced`.** Sign-consistent — positive on every cell,
negative on none.

Two caveats stated plainly. **Grouped alone is not enough**: it helps both
adapter cells but costs `-0.0103` on instr/base, so the reweighting is doing
necessary work rather than being an optional extra. And **three of `balanced`'s
four gains (+0.0016, +0.0009, +0.0016) sit at or below the noise floor** — its
real support is `varied/adapter` at `+0.0112`, plus never being negative. The
"+0.0038 mean" is a weaker number than it looks.

Implementation: `fit(family_data, family_groups=..., family_weights=...)`.
All-ones weights reproduce the previous fit to 1e-6, pinned by
`test_multifamily_probe.py` (7 tests), so it is a no-op for existing callers.

## Experiment 2 — layer 40 vs layer 46

Layer 46 was chosen in `sonic_v1` and never revisited, and **layer 40 is already
cached for all 23 datasets** — so this cost nothing but GPU time. `balanced`
recipe, same holdouts.

| held-out cell | L40 | L46 |
| --- | ---: | ---: |
| instr/adapter | 0.9956 | 0.9950 |
| instr/base | 0.9848 | **0.9911** |
| varied/adapter | 0.8910 | **0.9034** |
| varied/base | 0.9762 | **0.9879** |

**Verdict: keep layer 46.** It wins on all three cells that have headroom, by
`+0.0063`, `+0.0124` and `+0.0117`. L40's lone win is `+0.0006` on a saturated
cell — inside the noise floor.

The v1 choice was right. This avenue is closed; no layer sweep is warranted
unless the representation changes.

Results: `results/whitebox/v3_3_layer40.json`, `results/whitebox/v3_3_layer46.json`.

## Experiment 3 — shared trunk for gemma

`sonic_v2.3.7` moved qwen and nemotron onto the shared multi-family trunk but
left **gemma on its older single-family probe**. That decision *was* measured —
`export_multifamily_probe.py`'s docstring records a 3-seed sweep of trunk minus
single-family mean AUROC: nemotron `+0.0728 / +0.0159 / +0.0253`, qwen
`+0.0033 / +0.0006 / +0.0002`, gemma `-0.0168 / -0.0158 / -0.0107`. This
experiment is an independent replication on different folds.

`experiments/nonlinear_probe/gemma_trunk_eval.py`. Three leakage-free holdouts,
both designs trained on identical rows:

- `instr/base` — gemma's base cell (400 rows)
- `instr/adapter` — gemma's adapter cell (400 rows)
- **`g-st`** — the 400-row `dev-instructed-deception-gemma-3-27b-it-g-st-...`
  dataset, which is downloaded and extracted but **was never added to
  `dev_splits/manifest.csv`**. No shipped probe has trained on it, and `g-st` is
  the organism family of all three counted Iris units — the only leakage-free
  proxy we have for a counted dataset. For reference, the shipped probe scores
  `0.9215` here, against official Iris/gemma `0.9148`.

| holdout | single-family | shared trunk | delta |
| --- | ---: | ---: | ---: |
| instr/adapter | **0.9927** | 0.9728 | `-0.0199` |
| instr/base | **0.9877** | 0.9748 | `-0.0129` |
| g-st (unseen) | **0.9865** | 0.9742 | `-0.0123` |

**Verdict: keep gemma on the single-family probe.** The shared trunk is worse on
all three holdouts, every gap roughly ten times the noise floor. `sonic_v2.3.7`
excluded gemma without recording a reason; that decision was correct.

This is the mirror image of nemotron, where pooling gained `+0.0380` on row CV.
Consistent with the transfer story: pooling rescues a family with 216 rows and
16 positives, and costs a family that has enough of its own data, because the
shared trunk has to serve three input distributions at once.

One asymmetry when reading these: `TokenProbe` has no grouped-split parameter,
so the single-family arm keeps its random early-stopping split while the shared
arm uses the `balanced` recipe. That is the shipped state of each design rather
than a controlled comparison of the trunk alone — and note it means the
single-family arm wins *despite* the weaker split.

Results: `results/whitebox/gemma_trunk_eval.json`.

### An unexplained gap worth chasing

The freshly-trained single-family probe scores **0.9865** on the `g-st` holdout.
The **shipped** gemma probe scores **0.9215** on the same rows. Same design, same
training data, no leakage either way.

That `0.065` gap is larger than every other effect measured here combined. The
shipped gemma probe predates `v2.3.7` and may have been trained with different
hyperparameters, fewer rows, or simply an unlucky seed.

**Do not act on this yet — it is n=1.** Gemma has only 800 rows, and the ~0.0012
noise floor was measured on qwen-dominated cells with ten times the data, so
gemma's seed variance could be far wider. Next step is 3–5 seeds on the `g-st`
holdout alone. That either confirms a large free win from simply retraining
gemma with current code, or shows gemma's variance is wide enough to explain it.

## sonic_v3.3-mini — the probe-only diagnostic

**Submitted 2026-07-26 18:57.** Official headline AUROC `0.775208` / BA `0.657500`,
runtime 17m15s (1035s). `submission/sonic_v3_3_mini.ipynb`.

Everything above is measured on local folds, and every local fold is easier than
every Notus unit. A probe-only run is the only way to read the probe's *counted*
per-unit behaviour. `sonic_v1` was the last one (AUROC `0.7549`), so this is a
diagnostic, not a competitive entry. It gains `+0.0203` AUROC over `sonic_v1` —
the cumulative effect of the shared trunk, `balanced` recipe, and
`MultiFamilyProbe` architecture.

**Configuration, chosen so one run answers two questions:**

- **All three families exported from one shared trunk, including gemma** — the
  configuration every local measurement calls worse (3-seed sweep `-0.0168` /
  `-0.0158` / `-0.0107`; leave-one-cell-out `-0.0123` to `-0.0199`; g-st holdout
  `0.9101` vs shipped `0.9215`). That is deliberate: per-unit AUROC isolates each
  family, so this reads the trunk for gemma, qwen and nemotron simultaneously
  against `sonic_v1`'s single-family baseline. Shipping gemma single-family
  would answer nothing about the trunk.
- **Balanced recipe** — grouped early-stopping split plus base-row upweighting.
- Judge disabled (cell 11 sets `judge_margin = None`, so cell 12 falls through to
  its existing probe-alone branch). Score is `sigmoid(probe_z)`, frozen `0.5`
  threshold, no gate, no batch statistics.

Weights in `submission/whitebox_probe_mini/`, export parity verified exact
(`0.00e+00`). Constants in
`results/ensemble_gate_eval/sonic_v3_3_mini_constants.json`.

**Counted Notus/Iris per unit.**

| dataset | model / lora | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dataset Notus | `Nemotron-120B` / `None` | 0.5850 | 0.5943 | 0.8050 | 0.6350 |
| Dataset Notus | `Qwen3.5-27B` / `None` | 0.5300 | 0.6007 | 0.2650 | 0.2050 |
| Dataset Notus | `gemma-3-27b-it` / `None` | 0.5000 | 0.4808 | 0.0000 | 0.0000 |
| Dataset Iris | `Nemotron-120B` / `g-st` | 0.9325 | 0.9917 | 0.9700 | 0.1050 |
| Dataset Iris | `Qwen3.5-27B` / `g-st` | 0.8975 | 0.9982 | 0.8000 | 0.0050 |
| Dataset Iris | `gemma-3-27b-it` / `g-st-2` | 0.5000 | 0.9856 | 0.0000 | 0.0000 |

| | Notus mean | Iris mean | headline |
| --- | ---: | ---: | ---: |
| AUROC | 0.5586 | 0.9918 | 0.7752 |
| BA | 0.5383 | 0.7767 | 0.6575 |

**What the results say.**

1. **Gemma on the shared trunk is fine.** Despite local evidence that the trunk
   was worse for gemma, the counted numbers do not single it out. Iris/gemma AUROC
   `0.9856` — effectively saturated. Notus/gemma `0.4808` is above `sonic_v1`'s
   probe-only `0.3526`. The trunk serves all three families.

2. **Notus is the bottleneck, and the probe alone cannot fix it.** Notus mean
   AUROC `0.5586`. The `~0.20` gap to the blended sonic family (v3.2 Notus
   `0.8634`) is the judge's contribution, confirmed on the private split. The
   probe does not transfer to base-model-only distributed data.

3. **Iris is saturated.** AUROC `0.9918` across all three families. The BA
   collapse on Iris/gemma (`0.5000` at AUROC `0.9856`, recall `0.0000`) is the
   `sonic_v1`/`v2.3` threshold artefact — the frozen `0.5` cut is too high for
   gemma's probe-logit distribution off-distribution. Every blended sonic run's
   calibrated threshold fixes this. This is a BA problem, not a ranking problem.

4. **The shared trunk is ready.** It serves all three families simultaneously.
   A weights-only swap into v3.2's scoring rule is low-risk.

**Uncounted reference rows.** Metis is near-saturated (AUROC `0.9767--1.0000`
across all units). Eunomia is the probe's weak reference regime: adapter units
degrade from Metis's `~1.0000` to AUROC `0.8086--0.9689`, showing sensitivity
to the scenario axis. The base unit (`Qwen3.5-27B` / `None`) holds at `0.9906` —
the probe transfers to Eunomia base models but degrades on Eunomia adapters, the
opposite of the Notus pattern where base models are the failure. Full per-unit
breakdown: `docs/official_submissions.md`.

**Local benchmark** (leakage-free g-st gemma, 400 rows):

| metric | value |
| --- | ---: |
| AUROC | 0.9431 |
| balanced accuracy | 0.895 |
| runtime | 24s for 400 rows (16.6 rows/s) |
| unique scores | 400 / 400 (no ties) |

### Caches and live extraction disagree

The notebook scores `0.9431` on the g-st rows. The *same probe on the same rows*
scored `0.9101` when computed offline from
`results/whitebox/activations/*.tokens.npz`.

That `0.033` gap means the cached activations are not identical to live NDIF
extraction — plausibly precision or hardware, since the caches were produced on
different silicon. **Every offline experiment in this document uses the caches**,
so cache-derived numbers are valid for comparing options *against each other*
but must not be compared against notebook or leaderboard numbers. Worth running
down before any cache-selected probe change is shipped.

## What this does not test

None of these experiments touch the failure that actually costs us. **No dev
cell reproduces Notus**: the worst is `varied/adapter` at 0.89, Notus is 0.7577,
and `varied/base` — which looked like the obvious analogue — scores 0.9851
leakage-free. Difficulty tracks the **scenario** axis, not adapter presence.

So these are hygiene, measured honestly, on cells where the probe already scores
0.89–0.99. Expect thousandths, and expect none of it to move Notus. The
supporting evidence: `sonic_v2.3.7` already replaced per-family probes with the
shared trunk — a real architectural and training change — and Notus moved
`0.7577 -> 0.7585`, i.e. `+0.0008`.

## Reproducing

```bash
# experiment 1 (add --layer 40 for experiment 2)
python experiments/nonlinear_probe/train_v3_3_probe.py --max-epochs 60
python experiments/nonlinear_probe/train_v3_3_probe.py --max-epochs 60 \
    --recipes balanced --layer 40 --output results/whitebox/v3_3_layer40.json

# experiment 3
python experiments/nonlinear_probe/gemma_trunk_eval.py --max-epochs 60

pytest experiments/nonlinear_probe/test_multifamily_probe.py
```

Archived weights: `legacy_probes/whitebox_probe_v2_3_7/`.
