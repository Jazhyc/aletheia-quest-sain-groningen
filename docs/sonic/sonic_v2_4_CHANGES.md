# sonic_v2.4 — a draft that was never submitted

> **Status: abandoned. This version does not exist as a submission.**
>
> sonic_v2.4 was a set of ideas, not a release. It was never sent to the
> leaderboard. It has no score. The current version is **sonic_v3**; read
> `sonic_v3.md` instead.
>
> What happened to its two changes:
>
> - **The contrastive probe was built, measured and rejected.** It is worse than
>   the current probe on its own, and fusing the two gives `+0.014` on the worst
>   organism for a second forward pass and twice the activation download. See
>   `sonic_v3.md` section 7.2 and
>   `experiments/contrastive_probe/EVALUATION.md`.
> - **Spread-modulated gate trust was never measured.** It is still an untested
>   idea. It is not in sonic_v3.
>
> Keep this document for the reasoning behind those two ideas. Do not treat any
> number or instruction in it as current. The notebook it names now lives at
> `legacy_submissions/sonic_v2.4.ipynb`.

This document tells you what sonic_v2.4 would have added. Read the sonic_v2.3
document for the shared gate. You do not need the older documents.

This document uses Simplified Technical English. Sentences are short. Each
sentence gives one idea. The terms are constant: "the probe", "the judge", "the
contrastive probe", "the gate", "the score", "the threshold", "the spread", and
"the budget".

**In one line:** sonic_v2.4 keeps all of sonic_v2.3.5 and adds two changes. A
spread-modulated gate trust. An optional contrastive activation probe. Both
target the Notus datasets.

The submission notebook is `submission/sonic_v2.4.ipynb`.

---

## Why we made v2.4

The leaderboard score is the mean over the counted datasets. The counted
datasets are Notus and Iris.

- We are ahead on Iris.
- We are behind on Notus.
- The gap to the best team is Notus alone.

Notus uses the base model only. Notus uses no LoRA adapter. The probe is weak on
Notus. The judge is weak on Notus. The gap is both a ranking gap and a threshold
gap.

Two changes target the two parts of the gap.

1. The spread-modulated gate fixes a threshold problem. On datasets where the
   probe has no signal, the gate now leans toward the judge. The old gate could
   not tell which detector was better. The spread signal breaks the symmetry.
2. The contrastive probe targets the ranking gap. It reads a different signal
   from the model. The signal is the difference between activations with and
   without the deceptive prompt. This isolates deception from content.

---

## Change 1 — spread-modulated gate trust

The probe emits a score between 0 and 1 for each row. A strong probe spreads
its scores across the full range. A weak probe clusters its scores near 0.5.

The old gate used only rank-agreement to set the trust weight w. Agreement is
symmetric. The gate could not tell which detector was better.

The new gate measures the standard deviation of the probe scores. It scales the
trust weight by `min(1.0, std / 0.20)`. A probe with std < 0.20 gets less
trust. A probe with std ≥ 0.20 keeps full trust.

On Iris the probe has wide spread. The gate is unchanged. On Notus the probe
has narrow spread. The gate leans toward the judge.

This change is 12 lines in cell 13. It uses no new data. It uses no extra
compute.

---

## Change 2 — contrastive activation probe (optional)

The current probe reads absolute activations at decoder layer 46. The nonlinear
probe experiments proved this representation is maxed out. All architectures
plateau at the same ceiling.

The contrastive probe reads a different signal. It runs each conversation
through the model twice.

- Pass 1: the conversation as given (with deceptive system prompt).
- Pass 2: the same conversation with the system prompt removed.

It probes the *difference* between the two activation sets. The difference
cancels content-specific patterns. It isolates the instruction-driven deception
component.

The contrastive probe uses the same transformer token architecture as the
absolute probe. But it reads the difference signal, not the absolute signal. So
its weights are different.

The contrastive probe is optional. The notebook looks for weights at
`submission/whitebox_probe/{family}_probe/contrastive/`. When the weights are
present, the gate blends the contrastive probe as a third detector. When the
weights are absent, the gate falls through to the v2.3.5 probe-judge blend.

The contrastive probe adds one extra forward pass per batch. This doubles the
extraction time. The total runtime stays near the runtime of sonic_v2.3.5
because the extraction is the fast part of the run. The judge is the slow part.

See `experiments/contrastive_probe/` for the extraction and training pipeline.

---

## The gate with three detectors

The gate now accepts up to three detectors. The blend works in two stages.

1. **Base blend** — probe and judge blended as in v2.3.5. The spread-modulated
   trust weight replaces the old symmetric weight.
2. **Contrastive blend** — when the contrastive probe is present and has spread,
   it blends into the base. The contrastive weight grows when the base detectors
   disagree. The maximum contrastive weight is 0.3. The contrastive probe is
   oriented to match the base blend direction.

The threshold does not change. It still uses the judge's positive rate.

---

## What to expect

We have not tested the contrastive probe on real data. The probe weights do not
exist yet. You must run the extraction and training pipeline first.

The spread-modulated gate is safe. It changes only the trust weight. It uses a
real-time quality signal. It cannot collapse the decision column. It cannot
increase the probe trust. It can only shrink it when the probe has no signal.

On Iris the spread is wide. The gate is unchanged. On Notus the spread is
narrow. The gate leans toward the judge. The judge alone scores about AUROC
0.733 on Notus. The old gate scored about AUROC 0.690 on Notus Nemotron. The
new gate should recover the gap.

---

## How to test

Run these steps before you submit.

1. Run the dry pipeline with the judge forced on:

   ```
   SONIC_FORCE_JUDGE=1 python submit.py --dry
   ```

2. Check the telemetry line for each dataset. The `spread_qual` value should be
   near 1.0 on Iris datasets and near 0.3-0.7 on Notus datasets.
3. Compare the per-dataset AUROC against sonic_v2.3.5. The spread-modulated
   gate should match or improve v2.3.5 on every dataset.

For the contrastive probe:

1. Extract contrastive activations from all dev datasets.

   ```
   python experiments/contrastive_probe/extract.py --all-dev
   ```

2. Train the contrastive probe weights.

   ```
   python experiments/contrastive_probe/train.py
   ```

3. Run the dry pipeline again. The contrastive probe should activate.

   ```
   SONIC_FORCE_JUDGE=1 python submit.py --dry
   ```

4. Check the telemetry line. The `ctrv=True` flag shows the contrastive probe
   is active. The `ctrv_alpha` value shows its blend weight.

---

## What v2.4 does not change

- It does not change the absolute probe.
- It does not change the judge.
- It does not change cells 1 to 10.
- It does not change the threshold computation.
- It does not change the sonic_v2.3.5 fixes (median-split fallback, budget-aware
  judge retry).

---

## Rollback

The sonic_v2.3.5 notebook is in `legacy_submissions/sonic_v2.3.5.ipynb`. To roll back,
rename `sonic_v2.3.5.ipynb` to `sonic_v2.3.5.ipynb.bak` and rename the v2.4
notebook back. Or simply restore from git.
