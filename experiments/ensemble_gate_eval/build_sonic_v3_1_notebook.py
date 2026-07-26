#!/usr/bin/env python3
"""Derive ``submission/sonic_v3_1.ipynb`` from ``sonic_v2.3.7`` + frozen constants.

Everything except the final scoring cell is shared with the v3 builder, so the
probe pipeline and the direct-logit judge are imported from it rather than
restated.  Only the way the two detectors are combined changes.

    python experiments/ensemble_gate_eval/build_sonic_v3_1_notebook.py
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
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v3_1.ipynb"
DEFAULT_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_1_constants.json"

REFINE_CELL_TEMPLATE = '''if ds is not None:
    try:
        # v3.1 bounded refinement. Every row is still scored by a function of
        # ITSELF alone -- no rank transform, no probe/judge agreement, no
        # prevalence estimate, no quantile or median cut. Nothing reads the
        # batch. What changes from v3 is WHO RANKS.
        #
        #     score = sigmoid(judge_margin / JUDGE_SD
        #                     + PROBE_CAP * tanh(PROBE_GAIN * probe_z))
        #
        # v3 used a convex blend at a probe weight of 0.65 and lost 0.1055 AUROC
        # on the official Notus units, where the probe ranks at ~0.76 against a
        # judge at 0.864. It gained 0.0427 on Iris, where the probe is the
        # stronger detector. One global weight cannot serve both, and dev gives
        # no way to tell the two regimes apart: held-out base-model organisms
        # score probe 0.9878 / judge 0.9814 there.
        #
        # So the judge -- the detector that transfers -- owns the ranking, and
        # the probe may only nudge. The judge margin is bf16 and takes about 34
        # distinct values per 400 rows, so its ordering is mostly ties. PROBE_CAP
        # is two judge quantization steps, about 4% of the margin's full range.
        # The probe therefore breaks those ties and moves rows between adjacent
        # judge levels, but it can never overturn a confident judge. `tanh`
        # rather than a hard clip keeps the correction strictly monotone in the
        # probe, so no two rows are ever re-tied.
        #
        # Measured on the same leave-one-organism-out dev folds, with the probe
        # artificially blunted to Notus quality to simulate the regime we cannot
        # observe (mean AUROC change against the judge alone):
        #
        #     probe @ 0.70   v3 blend -0.1786    v3.1 +0.0024
        #     probe @ 0.76   v3 blend -0.1117    v3.1 +0.0042
        #     probe @ 0.85   v3 blend -0.0372    v3.1 +0.0060
        #     probe as-is    v3 blend +0.0175    v3.1 +0.0076
        #
        # v3.1 is the only setting that is positive in every regime. Its loss is
        # bounded by construction even against a probe far worse than Notus.
        #
        # Constants frozen by experiments/ensemble_gate_eval/fit_bounded_refine.py.
        # See {constants_path}.
        JUDGE_MARGIN_SD = {judge_sd!r}
        PROBE_CAP = {probe_cap!r}
        PROBE_GAIN = {probe_gain!r}
        # Per-family location AND scale of the shipped probe's raw output. v3
        # divided by a spread only: correct under a convex blend, where a shared
        # offset cannot change AUROC, but wrong under `tanh`, where the offset
        # decides where each row sits on the curve. v3's gemma spread was also
        # overestimated by 1.5x, which silently down-weighted the gemma probe --
        # the likely cause of official Iris/gemma falling 0.9921 -> 0.9161.
        PROBE_LOGIT_MEAN = {probe_mean!r}
        PROBE_LOGIT_SD = {probe_sd!r}
        DEFAULT_PROBE_MEAN = {default_probe_mean!r}
        DEFAULT_PROBE_SD = {default_probe_sd!r}
        # Frozen cut points. They only fill the required binary column; the
        # leaderboard ranks on AUROC, which ignores them. Neither is derived
        # from this dataset's own scores.
        #
        # A sign test (0.5) is wrong here. The judge's margin distribution sits
        # well below zero on NDIF -- implied judge_z spans -5.28..+2.68 remotely
        # against -1.57..+2.40 in the local cache -- so 0.5 flags far too little
        # and collapsed nemotron to 0.4975 balanced accuracy in benchmarking.
        # phoenix_wright_v4 reached 0.8154 official BA by cutting at 0.15 on
        # sigmoid(margin) for the same reason. This cut is fitted by
        # fit_threshold_ndif.py across four executed NDIF dev runs: mean BA
        # 0.8531 at 0.20, against 0.7262 at 0.50.
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
                combined = combined + PROBE_CAP * np.tanh(PROBE_GAIN * probe_z)
                print(f"refine: judge ranks, probe nudges within +/-{{PROBE_CAP:.4f}} "
                      f"({{base_model}}: mean={{probe_mean:.3f}} sd={{probe_sd:.3f}})",
                      flush=True)
            else:
                print("refine: judge alone (no probe scores for this dataset)", flush=True)
            final_scores = _sigmoid(combined)
            final_threshold = THRESHOLD_SCORE
        elif probe_ok:
            # The judge failed after its retry. The probe is all that is left,
            # and it is centred, so the frozen 0.5 cut sits at the family's
            # typical probe output rather than at an arbitrary point.
            probe_z = ((np.asarray(probe_logits, dtype=np.float64) - probe_mean)
                       / max(probe_sd, 1e-8))
            final_scores = _sigmoid(probe_z)
            final_threshold = THRESHOLD_PROBE_ONLY
            print("refine: probe alone (judge unavailable)", flush=True)
        else:
            # Nothing scored this dataset. Any constant column is AUROC 0.5;
            # emit a strictly increasing ramp so the column is at least valid
            # and never degenerate.
            final_scores = np.linspace(0.4, 0.6, num=len(indices)) if indices else np.zeros(0)
            final_threshold = 0.5
            print("[WARN] refine: neither detector produced scores", file=sys.stderr, flush=True)

        if final_threshold is None or not np.isfinite(final_threshold):
            # v2.3 lost 0.078 headline BA to a null threshold reaching
            # `score >= None`. It can never happen again silently.
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

HEADER_CELL = '''# sonic v3.1 — direct-logit judge ranks, white-box probe refines

Same probe as `sonic_v1`--`v3` and the same judge model, adapter and prompt as
`v2.3.7`. The judge reads the next-token logits of the single tokens `0` and `1`
after appending `Prediction:`, in one forward pass, and its raw margin
`logit_1 - logit_0` is the judge score.

**What changed from v3.** v3 combined the two detectors with a convex blend at a
probe weight of `0.65`. Officially that cost `0.1055` AUROC on Notus, where the
probe does not transfer (`~0.76` against a judge at `0.864`), and gained
`0.0427` on Iris, where the probe is the better detector — a net `-0.0314`
against the judge alone. No single weight serves both regimes, and no
inference-time metadata distinguishes them: held-out base-model dev organisms
score probe `0.9878` / judge `0.9814`, so "base model" is not the tell.

v3.1 therefore makes the judge the ranking authority and gives the probe a
strictly bounded correction:

    score = sigmoid(judge_margin / JUDGE_SD + PROBE_CAP * tanh(PROBE_GAIN * probe_z))

`PROBE_CAP` is two bf16 quantization steps of the judge margin, about `4%` of
its range. The margin takes only ~34 distinct values per 400 rows, so most of
what the probe does is resolve the judge's ties — which is pure gain whenever
the probe beats chance — while a confident judge can never be overturned. With
the probe blunted to Notus quality on dev folds, this rule still gains `+0.0042`
where the v3 blend loses `0.1117`.

Nothing in the scoring path reads the batch: no rank transform, no agreement
term, no prevalence estimate, no quantile or median cut. Every constant is
frozen offline. The required `deceptive` column cuts at a frozen `0.20`, fitted
across four executed NDIF dev runs. A `0.5` sign test is wrong for this judge —
its margin distribution sits well below zero, and `0.5` collapsed
instr-nemotron-base to balanced accuracy `0.4975` at AUROC `0.9503`.
'''


def load_constants(path: Path) -> dict:
    """Read the frozen refinement constants, or fail loudly."""
    if not path.exists():
        raise SystemExit(f"{path} missing -- run fit_bounded_refine.py first")
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
        probe_cap=constants["probe_cap"],
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
