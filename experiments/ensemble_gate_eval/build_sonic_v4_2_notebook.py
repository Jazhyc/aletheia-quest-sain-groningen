#!/usr/bin/env python3
"""Derive ``submission/sonic_v4_2.ipynb`` — dual-probe + judge-uncertainty gate.

Takes ``sonic_v4.ipynb`` as source.  Patches cell 12: on disagreement rows,
the cap opens when the judge is uncertain (|judge_z| < ~1.5).  Judge-confident
disagreement rows stay at BASE_CAP, preserving v4's Notus safety.

    OLD (v4):
        agreement = (combined * probe_z > 0).astype(np.float64)
        cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)

    NEW (v4.2):
        agreement = (combined * probe_z > 0).astype(np.float64)
        judge_uncertain = 1.0 / (1.0 + np.exp(np.abs(combined) - 1.5))
        cap = BASE_CAP + (agreement + (1.0 - agreement) * judge_uncertain) * (MAX_CAP - BASE_CAP)

    python experiments/ensemble_gate_eval/build_sonic_v4_2_notebook.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "submission/sonic_v4.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v4_2.ipynb"
EXPECTED_CELLS = 13
IDX_BLEND = 12

# ---- Replacements ------------------------------------------------------------

OLD_AGREEMENT_BLOCK = """                agreement = (combined * probe_z > 0).astype(np.float64)
                cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
                combined = combined + cap * PROBE_GAIN * probe_z
                mean_agreement = float(np.mean(agreement))
                print(f"refine: sign-gated probe nudge (dual-probe L40+L46, cap 4, linear) "
                      f"({base_model}: fused L40+L46 z-scores, "
                      f"agreement={mean_agreement:.3f} "
                      f"cap=[{BASE_CAP:.4f}, {MAX_CAP:.4f}])",
                      flush=True)"""

NEW_AGREEMENT_BLOCK = """                agreement = (combined * probe_z > 0).astype(np.float64)
                # v4.2: on disagreement rows, open the cap when the judge is
                # uncertain.  Judge-confident rows stay at BASE_CAP (v4 safety).
                # sigmoid(-|jz| + 1.5): ~1 when |jz|=0, ~0 when |jz|>3.
                judge_uncertain = 1.0 / (1.0 + np.exp(np.abs(combined) - 1.5))
                cap = BASE_CAP + (agreement + (1.0 - agreement) * judge_uncertain) * (MAX_CAP - BASE_CAP)
                combined = combined + cap * PROBE_GAIN * probe_z
                print(f"refine: uncertainty-gated probe nudge (dual-probe L40+L46, cap 4, linear) "
                      f"({base_model}: fused L40+L46 z-scores, "
                      f"agreement={float(np.mean(agreement)):.3f} "
                      f"mean_uncertain={float(np.mean(judge_uncertain)):.3f} "
                      f"cap=[{BASE_CAP:.4f}, {MAX_CAP:.4f}])",
                      flush=True)"""

OLD_RATIONALE = """        # v4: cell 10 fuses L40 and L46 probe z-scores into a single
        # pre-standardised probe_z (mean~0, sd~1).  The gate below is v3.8's,
        # unchanged: sign test, cap 4 steps, linear contribution."""

NEW_RATIONALE = """        # v4.2: dual probe (L40+L46) from v4, unchanged.  The gate adds
        # a judge-uncertainty exception to v4's sign test: on disagreement
        # rows, the cap opens when the judge is uncertain (|judge_z| < ~1.5).
        # Judge-confident disagreement rows stay at BASE_CAP (v4 safety).
        #
        #     agreement  = (judge_z * probe_z > 0)
        #     judge_uncertain = sigmoid(-|judge_z| + 1.5)
        #     cap = BASE_CAP + (agreement + (1-agreement)*judge_uncertain) * (MAX_CAP-BASE_CAP)"""

OLD_CONST_COMMENT = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP rolled back to 4 steps (v3.5's value),
        # tanh removed by test_linear_v3_7.py.  probe_z is pre-standardised
        # (cell 10 fuses L40+L46 z-scores).
        # See results/ensemble_gate_eval/sonic_v4_constants.json."""

NEW_CONST_COMMENT = """        # BASE_CAP and MAX_CAP frozen by fit_bounded_refine_v3_2.py,
        # tanh removed by test_linear_v3_7.py.  probe_z is pre-standardised
        # (cell 10 fuses L40+L46 z-scores).  v4.2 adds judge-uncertainty
        # gating on disagreement rows.
        # See results/ensemble_gate_eval/sonic_v4_constants.json."""


def patch(source: str, edits: list[tuple[str, str]], label: str) -> str:
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise SystemExit(f"{label}: expected 1 occurrence, found {count}")
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

    cells[IDX_BLEND]["source"] = patch(
        cells[IDX_BLEND]["source"],
        [
            (OLD_RATIONALE, NEW_RATIONALE),
            (OLD_CONST_COMMENT, NEW_CONST_COMMENT),
            (OLD_AGREEMENT_BLOCK, NEW_AGREEMENT_BLOCK),
        ],
        "cell 12")

    # Verify
    exec12 = "\n".join(l for l in cells[IDX_BLEND]["source"].splitlines()
                       if not l.strip().startswith("#"))
    if "np.tanh" in exec12:
        raise SystemExit("tanh survived")
    if "judge_uncertain" not in exec12:
        raise SystemExit("judge_uncertain missing")
    if "(combined * probe_z > 0)" not in exec12:
        raise SystemExit("sign test missing")
    if "1.0 - agreement" not in exec12 and "(1.0 - agreement)" not in exec12:
        raise SystemExit("disagreement branch missing")
    if "0.41675170554260993" not in exec12:
        raise SystemExit("MAX_CAP not 4 steps")

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
