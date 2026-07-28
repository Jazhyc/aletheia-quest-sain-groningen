#!/usr/bin/env python3
"""Derive ``submission/sonic_v4_3.ipynb`` — v4 with the judge standardisation removed.

Takes ``legacy_submissions/sonic_v4.ipynb`` as source.  Two edits, both in the
scoring path:

1. Cell 0  — the header describes v4.3.
2. Cell 12 — ``JUDGE_MARGIN_SD`` is deleted and the raw judge margin is scored
   directly.  The divisor is folded into the two caps and the threshold, so the
   ranking and the threshold decisions are unchanged row for row.

The probe keeps v4's frozen per-family standardisation in cell 10, untouched.
Cells 1–11 are asserted byte-identical to v4.  See docs/sonic/sonic_v4_3.md.

    python experiments/ensemble_gate_eval/build_sonic_v4_3_notebook.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v4.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v4_3.ipynb"
CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v4_3_constants.json"

EXPECTED_CELLS = 13
IDX_HEADER = 0
IDX_SCORE = 10
IDX_BLEND = 12
UNTOUCHED = tuple(range(1, 12))

# v4's constants, and the identity that removes the divisor.
JUDGE_MARGIN_SD = 1.199755138011975
V4_BASE_CAP = 0.20837585277130496
V4_MAX_CAP = 0.41675170554260993
V4_THRESHOLD = 0.2

BASE_CAP = V4_BASE_CAP * JUDGE_MARGIN_SD          # 0.25 exactly -- 2 judge steps
MAX_CAP = V4_MAX_CAP * JUDGE_MARGIN_SD            # 0.50 exactly -- 4 judge steps
THRESHOLD_SCORE = 1.0 / (1.0 + math.exp(-math.log(
    V4_THRESHOLD / (1.0 - V4_THRESHOLD)) * JUDGE_MARGIN_SD))

# ---- cell 0: header ----------------------------------------------------------

HEADER = """# sonic v4.3 -- v4 with the judge standardisation removed

Identical to sonic v4 in every measurable respect.  The judge's frozen
``JUDGE_MARGIN_SD`` is deleted and the raw margin is scored directly.

    v4    score = sigmoid(margin / 1.199755 + cap * probe_z)   caps 0.20838 / 0.41675, thr 0.2
    v4.3  score = sigmoid(margin             + cap * probe_z)   caps 0.25    / 0.50,    thr 0.15933

Dividing the judge by a positive constant is monotone, so it never calibrated
anything -- it only set how many judge units one probe sd buys.  That exchange
rate is the caps' job, so the divisor is folded into them and into the
threshold.  Both notebooks produce the same ranking and the same threshold
decisions, row for row, asserted by
``experiments/ensemble_gate_eval/test_sonic_v4_3_notebook.py``.

One judge step is 0.125 of raw margin, so in these units BASE_CAP is exactly
2 steps and MAX_CAP is exactly 4.

**Why remove it.** ``JUDGE_MARGIN_SD`` is the spread of the Phoenix 9B's margin
measured offline on dev.  A different judge model has no such measurement, so
any design that swaps the judge out inherits an unmeasured divisor.  Removing
it now means a new judge can be dropped in without one.

**What stays.** The dual probe at L40 and L46 and its frozen per-family
standardisation (cell 10), the judge and its prompt (cell 11), the sign test,
both caps, ``PROBE_GAIN``, and the probe-only fallback path.  The probe's
constants are kept deliberately: unlike the judge's margin, a probe logit has
no meaningful zero -- the per-family means are -0.4 (qwen), -2.5 (gemma) and
-6.3 (nemotron), and the sign test reads that zero.

**Known risks.** Unchanged from v4: the Notus proxy overestimates safety by
15x, part of the Iris shortfall is unexplained, the probe is in-sample on dev
folds.  v4.3 adds no risk of its own -- it is a numerical identity."""

# ---- cell 12: the gate, with the judge divisor folded into the caps ----------

BLEND_CELL = '''if ds is not None:
    try:
        # v3.2 agreement-modulated refinement.
        #
        # v3.1 used a fixed probe cap of 2 judge steps on every row.  That keeps
        # Notus safe (the probe does not transfer there and the fixed cap
        # prevents it from damaging the judge's ranking) but leaves Iris headroom
        # on the table -- the probe is near-perfect on Iris but can only nudge by
        # ~0.017 under the tight 2-step cap.  v2.3.5 scored Iris 0.9950 by giving
        # the probe near-full weight, but that design burned -0.1055 on Notus.
        #
        # v4.3 removes the judge's standardisation.  v4 scored
        # judge_z = judge_margin / JUDGE_MARGIN_SD, where JUDGE_MARGIN_SD
        # (1.199755) is the spread of the Phoenix 9B's margin measured offline on
        # dev.  Dividing by a positive constant is monotone, so it never
        # calibrated anything -- it only set how many judge units one probe sd
        # buys.  That exchange rate is the caps' job, so the divisor is folded
        # into them and into the threshold:
        #
        #     v4    margin / 1.199755 + cap * probe_z   caps 0.20838 / 0.41675, thr 0.2
        #     v4.3  margin            + cap * probe_z   caps 0.25    / 0.50,    thr 0.15933
        #
        # Same ranking, same threshold decisions, row for row.  In raw-margin
        # units one judge step is 0.125, so BASE_CAP is exactly 2 steps and
        # MAX_CAP exactly 4.
        #
        # The judge is the detector that can lose its divisor, and the probe is
        # not.  The judge's margin already has a meaningful zero -- logit("1") >
        # logit("0") is its own label boundary, which both the sign test and the
        # threshold read -- and a fixed scale, because it is one 9B with one
        # prompt on every row.  A probe logit has neither: the per-family means
        # are -0.4 (qwen), -2.5 (gemma), -6.3 (nemotron), so an uncentred probe
        # mis-signs whole families at the gate.  Cell 10 keeps v4's frozen
        # per-family standardisation for that reason.
        #
        # v4: cell 10 fuses L40 and L46 probe z-scores into a single
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
        #     agreement = (judge_margin * probe_z > 0)
        #     cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
        #     score = sigmoid(judge_margin + cap * PROBE_GAIN * probe_z)
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
        # provided everywhere.
        #
        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP rolled back to 4 steps (v3.5's value),
        # tanh removed by test_linear_v3_7.py.  probe_z is pre-standardised
        # (cell 10 fuses L40+L46 z-scores).
        # See results/ensemble_gate_eval/sonic_v4_3_constants.json.
        BASE_CAP = {base_cap!r}
        MAX_CAP = {max_cap!r}
        PROBE_GAIN = 1.0
        PROBE_LOGIT_MEAN = {{'qwen': -0.367495, 'gemma': -2.523992, 'nemotron': -6.329599}}
        PROBE_LOGIT_SD = {{'qwen': 6.626051, 'gemma': 4.510691, 'nemotron': 3.268104}}
        DEFAULT_PROBE_MEAN = -3.073695
        DEFAULT_PROBE_SD = 4.801615
        THRESHOLD_SCORE = {threshold!r}
        THRESHOLD_PROBE_ONLY = 0.5

        probe_mean = PROBE_LOGIT_MEAN.get(base_model, DEFAULT_PROBE_MEAN)
        probe_sd = PROBE_LOGIT_SD.get(base_model, DEFAULT_PROBE_SD)

        probe_ok = probe_logits is not None and len(probe_logits) == len(indices)
        judge_ok = judge_margin is not None and len(judge_margin) == len(indices)

        def _sigmoid(values):
            return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))

        if judge_ok:
            # v4.3: the raw margin, no divisor.  The caps carry the scale.
            combined = np.asarray(judge_margin, dtype=np.float64)
            if probe_ok:
                # v4: cell 10 fused L40+L46 z-scores are pre-standardised (mean~0, sd~1)
                probe_z = np.asarray(probe_logits, dtype=np.float64)
                # Per-row cap: open when judge and probe point the same way, tight otherwise.
                agreement = (combined * probe_z > 0).astype(np.float64)
                cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
                combined = combined + cap * PROBE_GAIN * probe_z
                mean_agreement = float(np.mean(agreement))
                print(f"refine: sign-gated probe nudge (dual-probe L40+L46, cap 4, linear) "
                      f"({{base_model}}: fused L40+L46 z-scores, raw judge margin, "
                      f"agreement={{mean_agreement:.3f}} "
                      f"cap=[{{BASE_CAP:.4f}}, {{MAX_CAP:.4f}}])",
                      flush=True)
            else:
                print("refine: judge alone (no probe scores for this dataset)", flush=True)
            final_scores = _sigmoid(combined)
            final_threshold = THRESHOLD_SCORE
        elif probe_ok:
            # v4: probe_z is pre-standardised by cell 10.  This branch never saw
            # the judge, so v4.3 leaves it and its 0.5 threshold alone.
            probe_z = np.asarray(probe_logits, dtype=np.float64)
            final_scores = _sigmoid(probe_z)
            final_threshold = THRESHOLD_PROBE_ONLY
            print("refine: probe alone (judge unavailable)", flush=True)
        else:
            final_scores = np.linspace(0.4, 0.6, num=len(indices)) if indices else np.zeros(0)
            final_threshold = 0.5
            print("[WARN] refine: neither detector produced scores", file=sys.stderr, flush=True)

        if final_threshold is None or not np.isfinite(final_threshold):
            raise ValueError(f"non-finite threshold {{final_threshold!r}}")

        util.write_submission(indices, final_scores, threshold=final_threshold)
    except Exception as _cell_err:
        print(f"[FATAL] refine/write failed: {{type(_cell_err).__name__}}: {{_cell_err}}",
              file=sys.stderr, flush=True)
        fallback = (np.asarray(judge_margin) if judge_margin is not None
                    else np.asarray(probe_logits) if probe_logits is not None
                    else np.zeros(len(indices)))
        util.write_submission(indices, 1.0 / (1.0 + np.exp(-np.clip(fallback, -80.0, 80.0))),
                              threshold=0.5)
else:
    util.write_submission([], np.zeros(0), threshold=0.5)
print("Done.")'''.format(base_cap=BASE_CAP, max_cap=MAX_CAP, threshold=THRESHOLD_SCORE)


def build(source: Path, output: Path) -> None:
    notebook = nbformat.read(source, as_version=4)
    if len(notebook.cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} cells in {source}, found {len(notebook.cells)}")

    original = [cell.source for cell in notebook.cells]

    if "JUDGE_MARGIN_SD" not in original[IDX_BLEND]:
        raise SystemExit("cell 12 of the source does not standardise the judge -- wrong source?")
    if "PROBE_LOGIT_MEAN_46" not in original[IDX_SCORE]:
        raise SystemExit("cell 10 of the source does not standardise the probe -- wrong source?")

    notebook.cells[IDX_HEADER].source = HEADER
    notebook.cells[IDX_BLEND].source = BLEND_CELL

    for index in UNTOUCHED:
        if notebook.cells[index].source != original[index]:
            raise SystemExit(f"cell {index} was modified but must stay byte-identical to v4")

    # The divisor is named in cell 12's comments, which is the point of them.
    # What must be gone is every executable mention of it.
    blend_code = "\n".join(line.split("#", 1)[0] for line in
                           notebook.cells[IDX_BLEND].source.splitlines())
    if "JUDGE_MARGIN_SD" in blend_code:
        raise SystemExit("the judge divisor survived into cell 12's code")
    if "PROBE_LOGIT_MEAN_46" not in notebook.cells[IDX_SCORE].source:
        raise SystemExit("the probe standardisation was lost from cell 10")

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"wrote {output} ({len(notebook.cells)} cells)")

    CONSTANTS.parent.mkdir(parents=True, exist_ok=True)
    CONSTANTS.write_text(json.dumps({
        "rule": "sigmoid(judge_margin + cap * probe_z) -- v4 with the judge divisor "
                "folded into the caps and the threshold",
        "derived_from": "results/ensemble_gate_eval/sonic_v4_constants.json",
        "judge_margin_sd_removed": JUDGE_MARGIN_SD,
        "base_cap": BASE_CAP,
        "max_cap": MAX_CAP,
        "probe_gain": 1.0,
        "threshold_score": THRESHOLD_SCORE,
        "threshold_probe_only": 0.5,
        "judge_step_in_raw_margin": BASE_CAP / 2.0,
        "probe_standardisation": "unchanged from v4 -- frozen per-family mean/sd in cell 10",
        "note": "a numerical identity with v4: same ranking, same threshold decisions",
    }, indent=1) + "\n")
    print(f"wrote {CONSTANTS}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    build(args.source, args.output)


if __name__ == "__main__":
    main()
