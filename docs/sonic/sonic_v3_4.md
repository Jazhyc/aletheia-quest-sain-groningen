# sonic v3.4

**Built 2026-07-27. Never submitted; overtaken by v3.5 on 2026-07-27 and moved
to `staged/sonic_v3_4.ipynb`.** It remains the only clean way to attribute the
v3.3 Iris loss to either MAX_CAP or the probe, so it is worth a later run.

v3.2 with the v3.3 probe. That is the whole change.

| | v3.2 | v3.3 | v3.4 |
| --- | ---: | ---: | ---: |
| the probe | per-family, gemma on its own | shared trunk, balanced | shared trunk, balanced |
| BASE_CAP | 0.208 (2 steps) | 0.104 (1 step) | 0.208 (2 steps) |
| MAX_CAP | 0.417 (4 steps) | 0.625 (6 steps) | 0.417 (4 steps) |
| official AUROC | 0.9031 | 0.9017 | — |

The judge is Phoenix v4 direct-logit. AGREEMENT_SCALE is 3.0. PROBE_GAIN is 1.0.
The threshold is 0.20. None of them moved. The scoring cell is byte-identical to
v3.2's except for the probe standardization constants, and cells 1–11 are
byte-identical to v3.2 outright. `test_sonic_v3_4_notebook.py` enforces both.

## Why this exists

v3.3 changed three things at once. The probe, BASE_CAP, and MAX_CAP. It lost
`-0.0014` headline AUROC. All of the loss was Iris (`0.9427 -> 0.9393`). Notus
improved to `0.8642`, the best of any sonic run.

Two causes are possible. The raised MAX_CAP, or the new probe. One run cannot
separate them. The per-unit pattern does not help: on Iris/gemma the cap never
opens, so the probe barely enters the score there, and both causes predict the
same shape.

v3.4 holds the caps at v3.2's fitted values and moves the probe alone.

## What the result will mean

| v3.4 Iris | reading | next step |
| --- | --- | --- |
| recovers toward 0.9427 | MAX_CAP at 6 steps caused the v3.3 loss | 6 steps is dead; ship v3.4 |
| stays near 0.9393 | the new probe caused it | revert to the v3.2 probe; 6 steps is still untested |

The headline follows Notus as well. v3.3's Notus `0.8642` came with BASE_CAP at
1 step. v3.4 puts BASE_CAP back to 2, so Notus should return to roughly v3.2's
`0.8634`. That is a loss of about `0.0008` against v3.3, and it is the price of
a clean read on Iris.

BASE_CAP at 1 step looked good on its own and is worth keeping later. It is not
kept here. Holding two constants and moving one thing is the point of the run.

## Standardization

PROBE_LOGIT_MEAN and PROBE_LOGIT_SD stay at v3.3's refitted values:

    PROBE_LOGIT_MEAN = {'qwen': -0.367495, 'gemma': -2.523992, 'nemotron': -6.329599}
    PROBE_LOGIT_SD   = {'qwen':  6.626051, 'gemma':  4.510691, 'nemotron':  3.268104}

These belong to the weights, not to the gate. The v3.2 values were fitted to the
old probe's logit scale and would be wrong here. Only the BA column depends on
them. AUROC is rank-based and the sigmoid is monotone.

## Build

```bash
python experiments/ensemble_gate_eval/build_sonic_v3_4_notebook.py
pytest experiments/ensemble_gate_eval/test_sonic_v3_4_notebook.py
```

Source notebook: `legacy_submissions/sonic_v3_3.ipynb`. Constants:
`results/ensemble_gate_eval/sonic_v3_4_constants.json`. Probe weights:
`submission/whitebox_probe/`, unchanged from the v3.3 run.
