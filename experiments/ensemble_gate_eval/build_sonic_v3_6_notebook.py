#!/usr/bin/env python3
"""Derive ``submission/sonic_v3_6.ipynb`` — v3.5 with MAX_CAP raised to 12 steps.

One constant changes. `MAX_CAP` goes from 4 judge steps (`0.41675`) to 12
(`1.25026`). The sign gate, `BASE_CAP`, `PROBE_GAIN`, the probe, the judge, the
prompt and the threshold are v3.5's, untouched.

Why: the cap bounds the probe's contribution in *absolute* judge-z units, so the
fraction of the probe's edge it lets through falls as that edge grows. Measured
capture at a +0.068 probe edge -- real Iris's condition -- is 69% at 4 steps
against 80% at 12. The Notus cost of the change is ~zero (`-0.0002` at the
measured 0.5586 probe quality) because the *direction test* carries the safety,
not the cap magnitude.

`fit_sign_gate_v3_5.py` could not see this: it varied the gate shape at fixed
caps of 2 and 4 steps and never raised `MAX_CAP`. Its "flat, always open" row
removes the direction test at cap 4, which is a different and much more dangerous
change. Selection and numbers: `test_cap_binding_v3_6.py`, constants in
`results/ensemble_gate_eval/sonic_v3_6_constants.json`.

    python experiments/ensemble_gate_eval/build_sonic_v3_6_notebook.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v3_5.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v3_6.ipynb"
DEFAULT_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_6_constants.json"

EXPECTED_CELLS = 13
SCORING_CELL = 12

HEADER_CELL = """# sonic v3.6 -- sign gate, cap raised to 12 judge steps

The probe, the judge, the prompt, the gate shape, `BASE_CAP`, `PROBE_GAIN` and
the threshold are `sonic_v3_5`'s, unchanged.  One constant differs:

    MAX_CAP = 12 judge steps   # v3.6  (1.25026)
    MAX_CAP =  4 judge steps   # v3.1 - v3.5  (0.41675)

    agreement = (judge_z * probe_z > 0)
    cap   = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
    score = sigmoid(judge_z + cap * tanh(PROBE_GAIN * probe_z))

**Why.**  The cap bounds the probe's contribution in absolute judge-z units, so
the share of the probe's edge it admits *falls as that edge grows*.  Measured
capture, sign gate held fixed, sweeping only the cap:

    probe edge   cap2   cap4   cap8  cap12  cap20  cap40
         0.025    82%    89%    96%   100%   104%   105%
         0.068    61%    69%    76%    80%    84%    86%   <- real Iris
         0.090    56%    64%    73%    78%    82%    83%

`sonic_v3_3_mini` measured the probe at AUROC 0.9918 on Iris against the judge's
0.9236, an edge of +0.068.  At 4 steps the cap is the binding constraint there.

**Why this is Notus-safe.**  The safety comes from the direction test, not from
the cap magnitude.  With the probe blunted to the 0.5586 `sonic_v3_3_mini`
measured on Notus, cap4 -> cap40 costs `-0.0003`; at 0.48 inverted, `-0.0003`.
v3.5's sweep never tested this because it varied the gate *shape* at caps fixed
to 2 and 4 steps.  Its "flat, always open" row removes the direction test, which
is a different change and is why that one costs `-0.0137` on Notus.

12 steps rather than 20 or 40: it takes 80% of the probe's edge against cap40's
86%, for a smaller excursion from the only cap regime validated on private data.

**Known risk.**  The local Notus proxy has the judge at 0.938 against real
Notus's ~0.864, so the near-zero Notus cost is measured on an easier regime than
the one that decides the headline.  And the harness says cap4 should capture 69%
on real Iris while `sonic_v3_5` actually captured 30.5%, so something beyond the
cap is also suppressing the probe there and the gain may not land in full.

Selection: `test_cap_binding_v3_6.py`; design note: `docs/sonic/sonic_v3_6.md`.

Nothing in the scoring path reads the batch: no rank transform, no prevalence
estimate, no quantile or median cut.  Every constant is frozen offline.
"""

CONSTANT_OLD = "        MAX_CAP = 0.41675170554260993\n"
CONSTANT_NEW = "        MAX_CAP = 1.2502551166278297\n"

RATIONALE_OLD = """        # v3.5 opens the cap on a direction test, not on a product:
"""
RATIONALE_NEW = """        # v3.6 keeps v3.5's direction test and raises MAX_CAP from 4 judge
        # steps to 12.  The cap bounds the probe in absolute judge-z units, so
        # the share of its edge that survives falls as that edge grows: at a
        # +0.068 probe edge -- what sonic_v3_3_mini measured on Iris, 0.9918
        # against the judge's 0.9236 -- capture is 69% at 4 steps and 80% at 12.
        # The direction test, not the cap size, is what keeps Notus safe: with
        # the probe blunted to the measured 0.5586, cap4 -> cap40 costs -0.0003.
        # Sweep: test_cap_binding_v3_6.py.
        #
        # v3.5 opens the cap on a direction test, not on a product:
"""

COMMENT_OLD = """        # Caps frozen by fit_bounded_refine_v3_2.py, gate shape by fit_sign_gate_v3_5.py.
        # See results/ensemble_gate_eval/sonic_v3_5_constants.json.
"""
COMMENT_NEW = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP by test_cap_binding_v3_6.py.
        # See results/ensemble_gate_eval/sonic_v3_6_constants.json.
"""

PRINT_OLD = '                print(f"refine: sign-gated probe nudge "\n'
PRINT_NEW = '                print(f"refine: sign-gated probe nudge (cap 12 steps) "\n'


def patch(source: str, edits: list[tuple[str, str]], where: str) -> str:
    for old, new in edits:
        if source.count(old) != 1:
            raise SystemExit(f"{where}: expected exactly one occurrence of\n{old!r}\n"
                             f"found {source.count(old)}")
        source = source.replace(old, new)
    return source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    args = parser.parse_args(argv)

    constants = json.loads(args.constants.read_text())
    expected = constants["max_cap"]
    if abs(expected - 1.2502551166278297) > 1e-12:
        raise SystemExit(f"constants file says MAX_CAP={expected}, builder writes 1.25026")
    if constants["base_cap"] != 0.20837585277130496:
        raise SystemExit("BASE_CAP must not move in v3.6")

    notebook = nbformat.read(args.source, as_version=4)
    cells = notebook["cells"]
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} cells in {args.source}, "
                         f"found {len(cells)}")

    cells[0]["source"] = HEADER_CELL
    cells[SCORING_CELL]["source"] = patch(
        cells[SCORING_CELL]["source"],
        [(CONSTANT_OLD, CONSTANT_NEW), (RATIONALE_OLD, RATIONALE_NEW),
         (COMMENT_OLD, COMMENT_NEW), (PRINT_OLD, PRINT_NEW)],
        f"cell {SCORING_CELL}")

    executable = "\n".join(line for line in cells[SCORING_CELL]["source"].splitlines()
                           if not line.strip().startswith("#"))
    if "0.41675170554260993" in executable:
        raise SystemExit("the old MAX_CAP survived the rewrite")
    if "AGREEMENT_SCALE" in executable:
        raise SystemExit("AGREEMENT_SCALE reappeared")

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
