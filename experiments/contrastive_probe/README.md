# Contrastive Activation Probe

Probes the *difference* between L46 activations with and without the deceptive
system prompt. The difference isolates the instruction-driven deception
component from content-specific patterns.

## Pipeline

1. **Extract** — `python experiments/contrastive_probe/extract.py --all-dev`
   Runs each dev dataset through the model twice (deceptive + neutral),
   saves per-token difference to `results/contrastive/activations/`.

2. **Train** — `python experiments/contrastive_probe/train.py`
   Loads all .npz files, trains one transformer token probe per family,
   saves weights to `submission/whitebox_probe/{family}_probe/contrastive/`.

3. **Evaluate** — `SONIC_FORCE_JUDGE=1 python submit.py --dry`
   The v2.4 notebook picks up contrastive weights automatically when present.
   Compare per-dataset AUROC/BA against the absolute probe baseline.

## Expected gains

The nonlinear probe experiments proved single-pass L46 is maxed out. The
contrastive difference should provide a genuinely orthogonal signal because
content-specific patterns cancel out. Expected AUROC gain on Notus: TBD
(requires extraction + training + dry run).
