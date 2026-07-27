#!/usr/bin/env python3
"""Derive ``submission/sonic_v3_7.ipynb`` — v3.6 with tanh squash removed.

Two things change from v3.6:
1. ``np.tanh(PROBE_GAIN * probe_z)`` → ``PROBE_GAIN * probe_z`` (linear probe contribution)
2. Header and rationale updated.

MAX_CAP stays at 12 steps. BASE_CAP, PROBE_GAIN, the probe, the judge, the
prompt and the threshold are v3.6's, untouched.

Why: the tanh compresses probe_z values above ~2. On Iris, where the probe
is strong (AUROC 0.9918), the most confident rows — the ones that should
dominate — are held back by the tanh asymptote. Removing it lets those
rows contribute at full strength. The sign gate still disables the probe
on disagreements, so Notus is protected. Measured:

    linear cap12 vs tanh cap12: Iris +0.0022, Notus −0.0002, headline +0.0010
    linear cap8  vs tanh cap12: Iris +0.0004, Notus −0.0002, headline +0.0000

So linear cap8 matches v3.6's gain with 4 fewer steps; linear cap12 adds
another +0.0010. Selection and numbers: ``test_linear_v3_7.py``.

    python experiments/ensemble_gate_eval/build_sonic_v3_7_notebook.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "submission/sonic_v3_6.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v3_7.ipynb"

EXPECTED_CELLS = 13
SCORING_CELL = 12

HEADER_CELL = """# sonic v3.7 -- sign gate, cap 12 steps, no tanh squash

The probe, the judge, the prompt, the gate shape, `BASE_CAP`, `MAX_CAP`,
`PROBE_GAIN` and the threshold are `sonic_v3_6`'s, unchanged.  One thing moves:

    # v3.6
    combined = combined + cap * np.tanh(PROBE_GAIN * probe_z)
    
    # v3.7
    combined = combined + cap * PROBE_GAIN * probe_z

    agreement = (judge_z * probe_z > 0)
    cap   = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
    score = sigmoid(judge_z + cap * probe_z)

**Why.** The tanh compresses ``probe_z`` above about ±2. On Iris, where the
probe is strong (AUROC 0.9918 measured by ``sonic_v3_3_mini``), the most
confident rows—the ones that should carry the ordering—are held back by the
tanh asymptote. Removing it lets them contribute at full strength.

**Measured (real Iris edge +0.068, Notus probe at 0.5586):**

    variant               Iris    Notus   headline  d_head vs tanh cap4
    tanh cap4  (v3.5)    0.9390  0.9206   0.9298   baseline
    tanh cap12 (v3.6)    0.9486  0.9204   0.9345   +0.0047
    linear cap8           0.9488  0.9202   0.9345   +0.0047  (same gain, 4 fewer steps)
    linear cap12 (v3.7)  0.9507  0.9202   0.9355   +0.0057  (+0.0010 over v3.6)

Linear cap8 matches v3.6's headline with a smaller excursion from the tested
cap regime. Linear cap12 adds another +0.0010.

**Why this is Notus-safe.** The sign gate, not the tanh, carries safety. On
disagreement rows the cap stays at BASE_CAP (0.208). Without tanh, the probe
can contribute up to 0.208 × |probe_z| instead of 0.208 × 1.0 — but at the
measured Notus quality of 0.5586, the cost is −0.0002, same as with tanh.

**Known risks.** The same ones as v3.6: (1) the local Notus proxy has the judge
at 0.938 against real Notus's ~0.864; (2) part of the Iris shortfall is
unexplained; (3) the probe is in-sample on dev folds.

Selection: ``test_linear_v3_7.py``; design note: ``docs/sonic/sonic_v3_7.md``.

Nothing in the scoring path reads the batch. Every constant is frozen offline.
"""

# v3.6 string to replace
TANH_OLD = "                combined = combined + cap * np.tanh(PROBE_GAIN * probe_z)\n"
TANH_NEW = "                combined = combined + cap * PROBE_GAIN * probe_z\n"

RATIONALE_OLD = """        # v3.6 keeps v3.5's direction test and raises MAX_CAP from 4 judge
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

RATIONALE_NEW = """        # v3.7 removes the tanh squash from v3.6's gate. The tanh compresses
        # probe_z above ~2, holding back the most confident rows on Iris where
        # the probe's strength (0.9918 AUROC) should dominate.  The sign gate
        # still carries Notus safety: on disagreement rows the cap stays at
        # BASE_CAP (0.208), and at the measured 0.5586 quality the cost of
        # removing the tanh is -0.0002.  Measured: linear cap12 gains +0.0010
        # headline over tanh cap12.  Sweep: test_linear_v3_7.py.
        #
        # v3.6 keeps v3.5's direction test and raises MAX_CAP from 4 to 12 steps.
        #
        # v3.5 opens the cap on a direction test, not on a product:
"""

COMMENT_OLD = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP by test_cap_binding_v3_6.py.
        # See results/ensemble_gate_eval/sonic_v3_6_constants.json.
"""
COMMENT_NEW = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP by test_cap_binding_v3_6.py,
        # tanh removed by test_linear_v3_7.py.
        # See results/ensemble_gate_eval/sonic_v3_7_constants.json.
"""

PRINT_OLD = '                print(f"refine: sign-gated probe nudge (cap 12 steps) "\n'
PRINT_NEW = '                print(f"refine: sign-gated probe nudge (cap 12 steps, linear) "\n'

# Also fix the gate formula in the comment
GATE_COMMENT_OLD = """        #     score = sigmoid(judge_z + cap * tanh(PROBE_GAIN * probe_z))
        #
        # v3.2 used clip(judge_z * probe_z / 3, 0, 1).  That opening"""
GATE_COMMENT_NEW = """        #     score = sigmoid(judge_z + cap * PROBE_GAIN * probe_z)
        #
        # v3.2 used clip(judge_z * probe_z / 3, 0, 1).  That opening"""


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
    args = parser.parse_args(argv)

    notebook = nbformat.read(args.source, as_version=4)
    cells = notebook["cells"]
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} cells in {args.source}, "
                         f"found {len(cells)}")

    cells[0]["source"] = HEADER_CELL
    cells[SCORING_CELL]["source"] = patch(
        cells[SCORING_CELL]["source"],
        [(TANH_OLD, TANH_NEW),
         (RATIONALE_OLD, RATIONALE_NEW),
         (COMMENT_OLD, COMMENT_NEW),
         (PRINT_OLD, PRINT_NEW),
         (GATE_COMMENT_OLD, GATE_COMMENT_NEW)],
        f"cell {SCORING_CELL}")

    executable = "\n".join(line for line in cells[SCORING_CELL]["source"].splitlines()
                           if not line.strip().startswith("#"))
    if "np.tanh" in executable:
        raise SystemExit("tanh survived the rewrite in executable code")
    if "AGREEMENT_SCALE" in executable:
        raise SystemExit("AGREEMENT_SCALE reappeared")
    if "0.41675170554260993" in executable:
        raise SystemExit("the old MAX_CAP (4 steps) reappeared")
    if "1.2502551166278297" not in executable:
        raise SystemExit("MAX_CAP is not 12 steps")

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
