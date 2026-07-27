#!/usr/bin/env python3
"""Derive ``submission/sonic_v3_4.ipynb`` — v3.2's gate with v3.3's probe.

`sonic_v3_3` changed three things at once: the probe weights, BASE_CAP (2 steps
-> 1) and MAX_CAP (4 steps -> 6). It scored 0.9017 against v3.2's 0.9031, and
the whole loss was Iris. The probe and the caps are confounded in that run, so
neither can be blamed.

v3.4 moves one thing. The caps go back to v3.2's fitted values and the probe
stays the v3.3 shared trunk. Everything else — judge, prompt, threshold,
AGREEMENT_SCALE, PROBE_GAIN — is untouched. The result attributes the v3.3
regression:

* Iris recovers toward 0.9427 -> MAX_CAP at 6 steps caused it, and 6 steps dies.
* Iris stays near 0.9393 -> the probe caused it, and 6 steps is still untested.

The probe's standardization constants travel with the weights, so they stay at
v3.3's refitted values. They are not part of the gate.

Built from `sonic_v3_3.ipynb`: cells 1-11 are copied untouched and only the two
cap constants in cell 12 change.

    python experiments/ensemble_gate_eval/build_sonic_v3_4_notebook.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v3_3.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "legacy_submissions/sonic_v3_4.ipynb"
DEFAULT_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_4_constants.json"

EXPECTED_CELLS = 13
SCORING_CELL = 12

FLOOR_COMMENT_OLD = "        # cap reverts to BASE_CAP -- a 1-step floor (halved from v3.2's 2-step)."
FLOOR_COMMENT_NEW = "        # cap reverts to BASE_CAP -- the same 2-step guarantee v3.1 provided."

HEADER_CELL = '''# sonic v3.4 -- v3.2's gate, v3.3's probe

The scoring rule, the judge, the prompt and the threshold are `sonic_v3_2`,
unchanged.  The probe weights are `sonic_v3_3`'s: all three families on one
shared trunk, trained with the balanced recipe (grouped early-stopping split
plus base-row upweighting).

    agreement = clip(judge_z * probe_z / AGREEMENT_SCALE, 0, 1)
    cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
    score = sigmoid(judge_z + cap * tanh(PROBE_GAIN * probe_z))

**Why this run exists.**  v3.3 changed the probe and both caps together and lost
0.0014 headline AUROC against v3.2, all of it on Iris (0.9427 -> 0.9393).  Notus
improved to 0.8642, the best of any sonic run.  Because three things moved at
once, the Iris loss could be the raised MAX_CAP or the new probe, and the
counted rows cannot separate them: on the one unit where the cap never opens
(Iris/gemma) the probe barely enters the score either, so both hypotheses
predict the same per-unit shape.

v3.4 holds the caps at v3.2's fitted values -- BASE_CAP = 2 judge quantization
steps, MAX_CAP = 4 -- and moves the probe alone.  If Iris returns toward 0.9427,
MAX_CAP at 6 steps caused the v3.3 regression.  If Iris stays near 0.9393, the
probe caused it.

The probe's standardization constants (PROBE_LOGIT_MEAN, PROBE_LOGIT_SD) belong
to the weights, not the gate, so they stay at v3.3's refitted values.  Only the
BA column depends on them; AUROC is rank-based and the sigmoid is monotone.

Nothing in the scoring path reads the batch: no rank transform, no prevalence
estimate, no quantile or median cut.  Every constant is frozen offline.
'''


def load_constants(path: Path) -> dict:
    """Read the frozen v3.4 constants.

    :param path: JSON file written alongside this builder
    :returns: the parsed constant block
    """
    if not path.exists():
        raise SystemExit(f"{path} missing")
    return json.loads(path.read_text())


def replace_assignment(source: str, name: str, value: object) -> str:
    """Rewrite one indented ``NAME = ...`` line in a notebook cell.

    :param source: the cell source
    :param name: the constant's name, assigned at 8-space indentation
    :param value: the new value, written with ``repr``
    :returns: the cell source with that one line replaced
    """
    marker = f"        {name} = "
    if marker not in source:
        raise SystemExit(f"cell {SCORING_CELL} has no {name} assignment")
    start = source.index(marker)
    end = source.index("\n", start)
    return source[:start] + f"{marker}{value!r}" + source[end:]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    args = parser.parse_args(argv)

    notebook = nbformat.read(args.source, as_version=4)
    cells = notebook["cells"]
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} cells in {args.source}, "
                         f"found {len(cells)}")

    constants = load_constants(args.constants)
    cells[0]["source"] = HEADER_CELL

    scoring = cells[SCORING_CELL]["source"]
    scoring = replace_assignment(scoring, "BASE_CAP", constants["base_cap"])
    scoring = replace_assignment(scoring, "MAX_CAP", constants["max_cap"])
    if FLOOR_COMMENT_OLD not in scoring:
        raise SystemExit(f"cell {SCORING_CELL} does not carry the v3.3 floor comment")
    scoring = scoring.replace(FLOOR_COMMENT_OLD, FLOOR_COMMENT_NEW)
    cells[SCORING_CELL]["source"] = scoring

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
