#!/usr/bin/env python3
"""Derive ``submission/sonic_v4_1.ipynb`` — dual-probe (L40+L46), confidence gate.

Takes ``sonic_v4.ipynb`` as source.  The ONLY change is in cell 12: the sign-test
agreement gate is replaced with a probe-confidence gate.

    OLD (v4, sign test):
        agreement = (combined * probe_z > 0)
        cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)

    NEW (v4.1, confidence gate):
        # Cap opens when the probe is confident, regardless of judge direction.
        # |probe_z| > CONFIDENCE_THRESHOLD -> cap near MAX_CAP
        # |probe_z| ~ 0 -> cap near BASE_CAP
        probe_confidence = sigmoid(|probe_z| - CONFIDENCE_THRESHOLD)
        cap = BASE_CAP + probe_confidence * (MAX_CAP - BASE_CAP)

Rationale: the sign test silences the probe on disagreement rows — exactly
where the probe is right and the judge is wrong on Iris.  A confidence gate
lets the probe through based on its own certainty.  On Notus the probe is
rarely confident (|z| rarely > 2), so the cap stays near BASE_CAP.  On Iris
the probe is routinely confident (|z| > 3-4), so the cap opens to MAX_CAP.
The judge-probe direction becomes irrelevant.

    python experiments/ensemble_gate_eval/build_sonic_v4_1_notebook.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v4.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v4_1.ipynb"
EXPECTED_CELLS = 13
IDX_BLEND = 12

# ---- Old cell 12 strings to replace -------------------------------------------

OLD_AGREEMENT_BLOCK = """                probe_z = np.asarray(probe_logits, dtype=np.float64)
                # Per-row cap: open when judge and probe point the same way, tight otherwise.
                agreement = (combined * probe_z > 0).astype(np.float64)
                cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
                combined = combined + cap * PROBE_GAIN * probe_z
                mean_agreement = float(np.mean(agreement))
                print(f"refine: sign-gated probe nudge (dual-probe L40+L46, cap 4, linear) "
                      f"({base_model}: fused L40+L46 z-scores, "
                      f"agreement={mean_agreement:.3f} "
                      f"cap=[{BASE_CAP:.4f}, {MAX_CAP:.4f}])",
                      flush=True)"""

NEW_AGREEMENT_BLOCK = """                probe_z = np.asarray(probe_logits, dtype=np.float64)
                # v4.1: confidence gate. Cap opens when the probe is confident,
                # regardless of whether the judge agrees.  The sign test (v4)
                # silenced the probe on disagreement rows, which on Iris are
                # exactly where the probe is right and the judge is wrong.
                # sigmoid(|z| - THRESHOLD): near 0 when uncertain, near 1 when
                # confident.  On Notus the probe is rarely confident (|z| < 2),
                # so the cap stays tight; on Iris it routinely exceeds 3-4.
                CONFIDENCE_THRESHOLD = 2.0
                probe_confidence = 1.0 / (1.0 + np.exp(-np.abs(probe_z) + CONFIDENCE_THRESHOLD))
                cap = BASE_CAP + probe_confidence * (MAX_CAP - BASE_CAP)
                combined = combined + cap * PROBE_GAIN * probe_z
                print(f"refine: confidence-gated probe nudge (dual-probe L40+L46, cap 4, linear) "
                      f"({base_model}: fused L40+L46 z-scores, "
                      f"mean_conf={float(np.mean(probe_confidence)):.3f}, "
                      f"cap=[{BASE_CAP:.4f}, {MAX_CAP:.4f}])",
                      flush=True)"""

# ---- Comment block replacements ------------------------------------------------

OLD_RATIONALE = """        # v4: cell 10 fuses L40 and L46 probe z-scores into a single
        # pre-standardised probe_z (mean~0, sd~1).  The gate below is v3.8's,
        # unchanged: sign test, cap 4 steps, linear contribution.
        #
        # Two probes at different layers have partially uncorrelated errors.
        # When both agree on a correction, the fused signal is stronger; when
        # they conflict, the average damps the contribution.  L40 was trained
        # with the same shared-trunk, multi-family, balanced recipe as L46.
        #
        # v3.8 keeps v3.7's linear (no-tanh) contribution but drops MAX_CAP
        # back to 4 steps (v3.5's value).
        #
        # v3.7 removes the tanh squash from v3.6's gate.
        #
        # v3.6 keeps v3.5's direction test and raises MAX_CAP from 4 to 12 steps.
        #
        # v3.5 opens the cap on a direction test, not on a product:
        #
        #     agreement = (judge_z * probe_z > 0)
        #     cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
        #     score = sigmoid(judge_z + cap * PROBE_GAIN * probe_z)
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

NEW_RATIONALE = """        # v4.1: confidence gate.  Dual-probe (L40+L46) from v4 kept unchanged;
        # cell 10 fuses z-scores.  The gate replaces v4's sign test with a
        # probe-confidence-based cap:
        #
        #     probe_confidence = sigmoid(|probe_z| - CONFIDENCE_THRESHOLD)
        #     cap = BASE_CAP + probe_confidence * (MAX_CAP - BASE_CAP)
        #     score = sigmoid(judge_z + cap * probe_z)
        #
        # When the probe is confident (|z| > THRESHOLD), the cap opens regardless
        # of what the judge thinks.  When it's uncertain (|z| ~ 0), the cap stays
        # near BASE_CAP.  On Notus the probe is rarely confident; on Iris it
        # routinely exceeds |z| > 3.  The judge-probe direction is irrelevant.
        #
        # v4 (sign test): cap = BASE_CAP when judge and probe disagree --
        # which on Iris is exactly where the probe is right and the judge wrong.
        # v4.1 fixes this by decoupling the cap from judge agreement."""

OLD_CONST_COMMENT = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP rolled back to 4 steps (v3.5's value),
        # tanh removed by test_linear_v3_7.py.  probe_z is pre-standardised
        # (cell 10 fuses L40+L46 z-scores).
        # See results/ensemble_gate_eval/sonic_v4_constants.json."""

NEW_CONST_COMMENT = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape adapted
        # to confidence gate (v4.1), MAX_CAP at 4 steps, tanh removed (v3.7).
        # probe_z is pre-standardised (cell 10 fuses L40+L46 z-scores).
        # See results/ensemble_gate_eval/sonic_v4_constants.json.
        # CONFIDENCE_THRESHOLD = 2.0 (fitted offline on dev folds)."""


def patch(source: str, edits: list[tuple[str, str]], label: str) -> str:
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise SystemExit(f"{label}: expected 1 occurrence of {old!r}, found {count}")
        source = source.replace(old, new)
    return source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    nb = nbformat.read(args.source, as_version=4)
    cells = nb["cells"]
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} cells, found {len(cells)}")

    # Patch cell 12 only -- everything else is identical to v4
    cells[IDX_BLEND]["source"] = patch(
        cells[IDX_BLEND]["source"],
        [
            (OLD_RATIONALE, NEW_RATIONALE),
            (OLD_CONST_COMMENT, NEW_CONST_COMMENT),
            (OLD_AGREEMENT_BLOCK, NEW_AGREEMENT_BLOCK),
        ],
        "cell 12")

    # Verify key invariants
    executable_12 = "\n".join(line for line in cells[IDX_BLEND]["source"].splitlines()
                              if not line.strip().startswith("#"))
    if "np.tanh" in executable_12:
        raise SystemExit("tanh survived")
    if "AGREEMENT_SCALE" in executable_12:
        raise SystemExit("AGREEMENT_SCALE reappeared")
    if "(combined * probe_z > 0)" in executable_12:
        raise SystemExit("sign test survived - agreement block not replaced")
    if "CONFIDENCE_THRESHOLD" not in executable_12:
        raise SystemExit("CONFIDENCE_THRESHOLD missing from executable code")
    if "probe_confidence" not in executable_12:
        raise SystemExit("probe_confidence missing from executable code")
    if "0.41675170554260993" not in executable_12:
        raise SystemExit("MAX_CAP not 4 steps")

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
