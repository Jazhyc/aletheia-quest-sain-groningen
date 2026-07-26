#!/usr/bin/env python3
"""Derive ``submission/sonic_v3_3_mini.ipynb`` — a probe-only diagnostic.

This is NOT a competitive entry. It removes the judge entirely so the counted
leaderboard rows report the **probe alone**, per unit, on the private splits.
`sonic_v1` was the last probe-only run (AUROC 0.7549 against v3_2's 0.9031), so
expect a headline in that region.

Why spend a run on it: every local fold is easier than every Notus unit, so the
probe's counted behaviour cannot be measured any other way. Two questions it
answers that nothing local can:

1. **Does the shared trunk help or hurt per family?** All three families are
   exported from one balanced-recipe trunk here, including **gemma**, which
   normally keeps its single-family probe because the trunk loses to it on every
   local fold (3-seed sweep: -0.0168 / -0.0158 / -0.0107; leave-one-cell-out:
   -0.0123 to -0.0199; g-st holdout 0.9101 vs the shipped 0.9215). Shipping the
   configuration local evidence calls worse is the point — per-unit AUROC
   isolates each family, so this reads the trunk for gemma, qwen and nemotron at
   once.
2. **What does the balanced training recipe do off-distribution?** Grouped
   early-stopping split plus base-row upweighting, measured at +0.0038 mean
   across dev cells, which is barely above the ~0.0012 noise floor.

Baseline caveat: `sonic_v1`'s per-unit *AUROC* is only partly recorded (the
metric was BA-first then) — Notus/gemma 0.3526 and Iris/gemma 0.9911 are the
anchors. Read absolute numbers, not clean deltas.

Built from `sonic_v3_2.ipynb`'s pipeline: cells 1-10 (probe) are unchanged
except for the weights directory, cell 11 (judge) is disabled, and cell 12 falls
through to its existing probe-alone branch.

    python experiments/ensemble_gate_eval/build_sonic_v3_3_mini_notebook.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "submission/sonic_v3_2.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v3_3_mini.ipynb"
DEFAULT_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_3_mini_constants.json"
PROBE_SUBDIR = "whitebox_probe_mini"

HEADER_CELL = '''# sonic v3.3-mini -- probe-only diagnostic

**This is a diagnostic, not a competitive entry.** The judge is removed, so the
counted rows report the *probe alone* on the private splits. `sonic_v1` was the
last probe-only run and scored AUROC `0.7549` against `sonic_v3_2`'s `0.9031`.

Every local fold is easier than every Notus unit, so the probe's counted
per-unit behaviour cannot be measured any other way. This run answers two
questions no offline experiment can:

1. **Shared trunk, per family.** All three families come from one shared trunk
   here, including **gemma** -- which normally keeps its single-family probe
   because the trunk loses on every local fold (3-seed sweep `-0.0168` /
   `-0.0158` / `-0.0107`; leave-one-cell-out `-0.0123` to `-0.0199`; leakage-free
   g-st holdout `0.9101` vs the shipped `0.9215`). Shipping the configuration
   local evidence calls worse is deliberate: per-unit AUROC isolates each
   family, so one run reads the trunk for gemma, qwen and nemotron at once.
2. **The balanced training recipe off-distribution.** Grouped early-stopping
   split (whole organisms held out, so model selection rewards transfer rather
   than memorising the training organisms) plus base-model rows upweighted from
   17% of the corpus to ~50% of the loss. Measured at `+0.0038` mean across dev
   cells, against a `~0.0012` run-to-run noise floor.

Scoring is `sigmoid(probe_z)` with a frozen `0.5` threshold. No judge, no gate,
no batch statistics. AUROC is rank-based and the sigmoid is monotone, so the
standardization constants affect the balanced-accuracy column only.

Probe weights: `submission/whitebox_probe_mini/`, one shared trunk fitted on all
8,216 dev rows with the balanced recipe, split into per-family weights (export
parity verified exact).
'''

JUDGE_CELL = '''# v3.3-mini: the judge is deliberately disabled.
#
# This run measures the probe alone on the counted units.  Cell 12 already has a
# probe-only branch for the case where the judge is unavailable, so setting
# judge_margin to None here routes scoring through it without touching the
# blend logic that sonic_v3_2 ships.
#
# Everything else in the pipeline -- extraction, pooling, standardization -- is
# byte-identical to sonic_v3_2, so any difference in the counted rows is
# attributable to the probe weights and the missing judge, nothing else.
judge_margin = None
print("v3.3-mini: judge disabled; scoring the probe alone", flush=True)
'''


def load_constants(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} missing")
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    args = parser.parse_args(argv)

    notebook = nbformat.read(args.source, as_version=4)
    cells = notebook["cells"]
    if len(cells) != 13:
        raise SystemExit(f"expected 13 cells in {args.source}, found {len(cells)}")

    constants = load_constants(args.constants)

    cells[0]["source"] = HEADER_CELL

    # Point the probe loader at the mini weights.
    old_dir = 'submission/whitebox_probe/{base_model}_probe'
    new_dir = f'submission/{PROBE_SUBDIR}/{{base_model}}_probe'
    if old_dir not in cells[3]["source"]:
        raise SystemExit("cell 3 does not contain the expected probe_dir literal")
    cells[3]["source"] = cells[3]["source"].replace(old_dir, new_dir)

    cells[11]["source"] = JUDGE_CELL

    # Refit standardization for the new weights. Only the BA column depends on
    # these -- AUROC is rank-based and the sigmoid is monotone.
    scoring = cells[12]["source"]
    for name, key in (("PROBE_LOGIT_MEAN", "probe_logit_mean"),
                      ("PROBE_LOGIT_SD", "probe_logit_sd")):
        start = scoring.index(f"        {name} = ")
        end = scoring.index("\n", start)
        scoring = scoring[:start] + f"        {name} = {constants[key]!r}" + scoring[end:]
    for name, key in (("DEFAULT_PROBE_MEAN", "default_probe_mean"),
                      ("DEFAULT_PROBE_SD", "default_probe_sd")):
        start = scoring.index(f"        {name} = ")
        end = scoring.index("\n", start)
        scoring = scoring[:start] + f"        {name} = {constants[key]!r}" + scoring[end:]
    cells[12]["source"] = scoring

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
