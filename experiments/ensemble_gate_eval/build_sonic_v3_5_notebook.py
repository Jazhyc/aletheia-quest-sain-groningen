#!/usr/bin/env python3
"""Derive ``submission/sonic_v3_5.ipynb`` — v3.4 with a sign gate.

One line of scoring changes. v3.2's cap opening

    agreement = clip(judge_z * probe_z / AGREEMENT_SCALE, 0, 1)

becomes a direction test

    agreement = (judge_z * probe_z > 0)

and AGREEMENT_SCALE disappears. Caps, probe, judge, prompt and threshold are
v3.4's, untouched.

Why: the product's opening correlates +0.94 with `|judge_z|`, so the cap is
widest where the judge is already certain and near-shut where it is unsure. In
the noise regime that is a safety feature -- confident rows cannot be reordered
by a random probe -- and it is why flat and judge-uncertainty gates lose badly
when the probe is blunted to Notus quality. But on folds where the probe
outclasses the judge, which is Iris's condition, the same coupling withholds the
probe exactly where it would pay: the product captures 54% of the probe's edge
against the sign test's 66%. The sign test keeps the direction check and drops
the magnitude coupling. Selection and numbers: `fit_sign_gate_v3_5.py`.

v3.5 holds `submission/` as of 2026-07-27; `sonic_v3_4.ipynb` moved to `legacy_submissions/`
unsubmitted, since the runner accepts exactly one notebook. v3.4 still answers
whether v3.3's Iris loss was the probe or MAX_CAP, so it is worth a later run,
but v3.5 supersedes it as a candidate.

    python experiments/ensemble_gate_eval/build_sonic_v3_5_notebook.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v3_4.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v3_5.ipynb"
DEFAULT_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_5_constants.json"

EXPECTED_CELLS = 13
SCORING_CELL = 12

RATIONALE_OLD = """        # v3.2 makes the cap per-row: large when the judge and probe agree on
        # direction, tight when they disagree or the judge is uncertain:
        #
        #     agreement = clip(judge_z * probe_z / AGREEMENT_SCALE, 0, 1)
        #     cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
        #     score = sigmoid(judge_z + cap * tanh(PROBE_GAIN * probe_z))
        #
        # On Iris, both detectors point the same way for deceptive rows --
        # agreement is high, cap opens to MAX_CAP, and the probe recovers most of
        # its v2.3.5 Iris ranking.  On Notus, the probe is near-random --
        # agreement is noise, cap stays close to BASE_CAP, and Notus stays safe.
        #
        # When judge and probe disagree (opposite signs), agreement = 0 and the
        # cap reverts to BASE_CAP -- the same 2-step guarantee v3.1 provided."""

RATIONALE_NEW = """        # v3.5 opens the cap on a direction test, not on a product:
        #
        #     agreement = (judge_z * probe_z > 0)
        #     cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
        #     score = sigmoid(judge_z + cap * tanh(PROBE_GAIN * probe_z))
        #
        # v3.2 used clip(judge_z * probe_z / 3, 0, 1).  That opening
        # correlates +0.94 with |judge_z|, so the cap was widest where the judge was
        # already certain and nearly shut where it was unsure.  When the probe is
        # noise that is protective -- confident rows sit far from the ordering
        # boundary, so a random probe cannot reorder them.  But where the probe
        # outclasses the judge, which is Iris, it withholds the probe exactly where
        # it would pay: the product captures 54% of the probe's edge over the judge
        # on such folds, the sign test 66%.
        #
        # The direction check is what carries the safety, so it stays.  Disagreeing
        # rows (opposite signs) still get BASE_CAP -- the 2-step guarantee v3.1
        # provided everywhere."""

CONSTANT_OLD = """        # Constants frozen by fit_bounded_refine_v3_2.py.  See results/ensemble_gate_eval/sonic_v3_2_constants.json.
        JUDGE_MARGIN_SD = 1.199755138011975
        BASE_CAP = 0.20837585277130496
        MAX_CAP = 0.41675170554260993
        AGREEMENT_SCALE = 3.0
        PROBE_GAIN = 1.0"""

CONSTANT_NEW = """        # Caps frozen by fit_bounded_refine_v3_2.py, gate shape by fit_sign_gate_v3_5.py.
        # See results/ensemble_gate_eval/sonic_v3_5_constants.json.
        JUDGE_MARGIN_SD = 1.199755138011975
        BASE_CAP = 0.20837585277130496
        MAX_CAP = 0.41675170554260993
        PROBE_GAIN = 1.0"""

GATE_OLD = """                # Per-row cap: large when judge and probe agree, tight otherwise.
                raw_agreement = combined * probe_z / max(AGREEMENT_SCALE, 1e-8)
                agreement = np.clip(raw_agreement, 0.0, 1.0)"""

GATE_NEW = """                # Per-row cap: open when judge and probe point the same way, tight otherwise.
                agreement = (combined * probe_z > 0).astype(np.float64)"""

PRINT_OLD = '                print(f"refine: agreement-modulated probe nudge "'
PRINT_NEW = '                print(f"refine: sign-gated probe nudge "'

HEADER_CELL = '''# sonic v3.5 -- sign-gated probe refinement

The probe, the judge, the prompt, the caps and the threshold are `sonic_v3_4`'s,
unchanged.  One line of the scoring rule differs:

    agreement = (judge_z * probe_z > 0)        # v3.5
    agreement = clip(judge_z * probe_z / 3, 0, 1)   # v3.2 - v3.4

    cap   = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
    score = sigmoid(judge_z + cap * tanh(PROBE_GAIN * probe_z))

**Why.**  The product opening correlates +0.94 with `|judge_z|`.  Sorted by judge
confidence, the least certain fifth of rows opened the cap 3% and the most certain
fifth opened it 78%.  So the probe got its loudest vote where the judge was
already sure -- rows far from the ordering boundary, where nothing can be
reordered -- and was muted where the judge was undecided.

When the probe is noise that coupling protects us, which is why flat and
judge-uncertainty gates collapse once the probe is blunted to the AUROC 0.5586
`sonic_v3_3_mini` measured on Notus.  The direction check is doing that work, not
the magnitude.  On folds where the probe outclasses the judge -- Iris's condition
-- the magnitude term only costs: the product captures 54% of the probe's edge,
the sign test 66%.

Selected against the probe quality Notus actually has (0.56, and 0.48 inverted),
not the 0.76 the v3.2 sweep assumed.  Full selection: `fit_sign_gate_v3_5.py`;
design note: `docs/sonic/sonic_v3_5.md`.

Nothing in the scoring path reads the batch: no rank transform, no prevalence
estimate, no quantile or median cut.  Every constant is frozen offline.
'''


def patch(source: str, replacements: list[tuple[str, str]], label: str) -> str:
    """Apply exact-text replacements, refusing to guess if any anchor is missing.

    :param source: cell source to rewrite
    :param replacements: (old, new) pairs, each of which must appear exactly once
    :param label: cell name used in error messages
    :returns: the rewritten source
    """
    for old, new in replacements:
        if source.count(old) != 1:
            raise SystemExit(f"{label}: anchor appears {source.count(old)} times, expected 1:\n"
                             f"{old[:120]}...")
        source = source.replace(old, new)
    return source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    args = parser.parse_args(argv)

    if not args.constants.exists():
        raise SystemExit(f"{args.constants} missing -- run fit_sign_gate_v3_5.py first")
    constants = json.loads(args.constants.read_text())
    if constants.get("gate") != "sign":
        raise SystemExit(f"{args.constants} does not describe the sign gate")

    notebook = nbformat.read(args.source, as_version=4)
    cells = notebook["cells"]
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} cells in {args.source}, "
                         f"found {len(cells)}")

    cells[0]["source"] = HEADER_CELL
    cells[SCORING_CELL]["source"] = patch(
        cells[SCORING_CELL]["source"],
        [(RATIONALE_OLD, RATIONALE_NEW), (CONSTANT_OLD, CONSTANT_NEW),
         (GATE_OLD, GATE_NEW), (PRINT_OLD, PRINT_NEW)],
        f"cell {SCORING_CELL}")

    # The constant must be gone from the code; comments may still name it.
    executable = "\n".join(line for line in cells[SCORING_CELL]["source"].splitlines()
                           if not line.strip().startswith("#"))
    if "AGREEMENT_SCALE" in executable:
        raise SystemExit("AGREEMENT_SCALE survived the rewrite")

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
