#!/usr/bin/env python3
"""Derive ``submission/sonic_v3_2.ipynb`` from ``sonic_v2.3.7`` + frozen constants.

Same probe pipeline and direct-logit judge as v3.1.  Only the scoring rule
changes: v3.1's fixed 2-step cap becomes a per-row cap modulated by
judge/probe agreement.

    python experiments/ensemble_gate_eval/build_sonic_v3_2_notebook.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sonic_v3_notebook import (  # noqa: E402
    JUDGE_CELL,
    PROBE_INIT_PATCHES,
    PROBE_LOGIT_PATCHES,
    SPLIT_GATE_ANCHOR,
    SPLIT_GATE_REPLACEMENT,
    patch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v2.3.7.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v3_2.ipynb"
DEFAULT_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_2_constants.json"

REFINE_CELL_TEMPLATE = '''if ds is not None:
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
        # v3.2 makes the cap per-row: large when the judge and probe agree on
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
        # cap reverts to BASE_CAP -- the same 2-step guarantee v3.1 provided.
        #
        # Constants frozen by fit_bounded_refine_v3_2.py.  See {constants_path}.
        JUDGE_MARGIN_SD = {judge_sd!r}
        BASE_CAP = {base_cap!r}
        MAX_CAP = {max_cap!r}
        AGREEMENT_SCALE = {agreement_scale!r}
        PROBE_GAIN = {probe_gain!r}
        PROBE_LOGIT_MEAN = {probe_mean!r}
        PROBE_LOGIT_SD = {probe_sd!r}
        DEFAULT_PROBE_MEAN = {default_probe_mean!r}
        DEFAULT_PROBE_SD = {default_probe_sd!r}
        THRESHOLD_SCORE = {threshold_score!r}
        THRESHOLD_PROBE_ONLY = {threshold_probe!r}

        probe_mean = PROBE_LOGIT_MEAN.get(base_model, DEFAULT_PROBE_MEAN)
        probe_sd = PROBE_LOGIT_SD.get(base_model, DEFAULT_PROBE_SD)

        probe_ok = probe_logits is not None and len(probe_logits) == len(indices)
        judge_ok = judge_margin is not None and len(judge_margin) == len(indices)

        def _sigmoid(values):
            return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))

        if judge_ok:
            combined = np.asarray(judge_margin, dtype=np.float64) / JUDGE_MARGIN_SD
            if probe_ok:
                probe_z = ((np.asarray(probe_logits, dtype=np.float64) - probe_mean)
                           / max(probe_sd, 1e-8))
                # Per-row cap: large when judge and probe agree, tight otherwise.
                raw_agreement = combined * probe_z / max(AGREEMENT_SCALE, 1e-8)
                agreement = np.clip(raw_agreement, 0.0, 1.0)
                cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
                combined = combined + cap * np.tanh(PROBE_GAIN * probe_z)
                mean_agreement = float(np.mean(agreement))
                print(f"refine: agreement-modulated probe nudge "
                      f"({{base_model}}: mean={{probe_mean:.3f}} sd={{probe_sd:.3f}}, "
                      f"agreement={{mean_agreement:.3f}} "
                      f"cap=[{{BASE_CAP:.4f}}, {{MAX_CAP:.4f}}])",
                      flush=True)
            else:
                print("refine: judge alone (no probe scores for this dataset)", flush=True)
            final_scores = _sigmoid(combined)
            final_threshold = THRESHOLD_SCORE
        elif probe_ok:
            probe_z = ((np.asarray(probe_logits, dtype=np.float64) - probe_mean)
                       / max(probe_sd, 1e-8))
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
print("Done.")
'''

HEADER_CELL = '''# sonic v3.2 -- agreement-modulated probe refinement

Same probe as `sonic_v1`--`v3.1` and the same judge model, adapter and prompt as
`v2.3.7`.  The judge reads the next-token logits of the single tokens `0` and `1`
after appending `Prediction:`, in one forward pass, and its raw margin
`logit_1 - logit_0` is the judge score.

**What changed from v3.1.**  v3.1 capped the probe at a fixed 2 judge
quantization steps on every row.  That kept Notus safe (flat against the judge
alone) but left Iris headroom unused -- the probe is near-perfect on Iris
(0.9950 in v2.3.5) yet could only add ~0.017 AUROC under the tight fixed cap.

v3.2 makes the cap per-row: large when the judge and probe agree on direction,
tight when they disagree or the judge is uncertain:

    agreement = clip(judge_z * probe_z / AGREEMENT_SCALE, 0, 1)
    cap = BASE_CAP + agreement * (MAX_CAP - BASE_CAP)
    score = sigmoid(judge_z + cap * tanh(PROBE_GAIN * probe_z))

On Iris, both detectors point the same way for deceptive rows -- agreement is
high, the cap opens toward MAX_CAP, and the probe recovers most of its Iris
ranking.  On Notus, the probe is near-random -- agreement is noise, the cap
stays close to BASE_CAP, and Notus stays safe.  When judge and probe disagree
(opposite signs), agreement = 0 and the cap reverts to BASE_CAP -- the same
2-step guarantee v3.1 provided everywhere.

BASE_CAP, MAX_CAP, and AGREEMENT_SCALE are selected by sweeping 15 combinations
across 20 leave-one-organism-out dev folds at 4 probe quality levels, choosing
the combination that maximizes worst-fold mean AUROC at Notus probe quality
(0.76).  See `fit_bounded_refine_v3_2.py`.

Nothing in the scoring path reads the batch: no rank transform, no prevalence
estimate, no quantile or median cut.  Every constant is frozen offline.
'''


def load_constants(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} missing -- run fit_bounded_refine_v3_2.py first")
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

    cells[0]["source"] = HEADER_CELL
    cells[1]["source"] = (
        cells[1]["source"].split(SPLIT_GATE_ANCHOR)[0] + SPLIT_GATE_REPLACEMENT
    )
    cells[3]["source"] = patch(cells[3]["source"], PROBE_INIT_PATCHES, "cell 3")
    cells[10]["source"] = patch(cells[10]["source"], PROBE_LOGIT_PATCHES, "cell 10")

    constants = load_constants(args.constants)
    cells[11]["source"] = JUDGE_CELL
    cells[12]["source"] = REFINE_CELL_TEMPLATE.format(
        constants_path=args.constants.relative_to(REPO_ROOT),
        judge_sd=constants["judge_margin_sd"],
        base_cap=constants["base_cap"],
        max_cap=constants["max_cap"],
        agreement_scale=constants["agreement_scale"],
        probe_gain=constants["probe_gain"],
        probe_mean=constants["probe_logit_mean"],
        probe_sd=constants["probe_logit_sd"],
        default_probe_mean=constants["default_probe_mean"],
        default_probe_sd=constants["default_probe_sd"],
        threshold_score=constants["threshold_score"],
        threshold_probe=constants["threshold_probe_only"],
    )

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None
    nbformat.write(notebook, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
