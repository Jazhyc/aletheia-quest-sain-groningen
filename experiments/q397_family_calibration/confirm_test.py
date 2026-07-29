#!/usr/bin/env python3
"""Apply the validation-frozen family calibration once on local test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.q397_family_calibration.calibrate import (
    apply_parameters,
    load_intent_scores,
    load_jsonl,
    overlay_intent,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = (
    ROOT / "results/blackbox/q397_family_calibration_v1/summary.json"
)
DEFAULT_TEST = (
    ROOT / "results/blackbox/q397_family_calibration_test_cache_v1/scores.jsonl"
)
DEFAULT_INTENT = (
    ROOT
    / "results/blackbox/specialist_necessity_ndif_v1/"
    "intent_test_current_rank1.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "results/blackbox/q397_family_calibration_test_confirmation_v1"
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Confirm only the train-selected candidate that passed validation."""
    calibration = json.loads(args.calibration_summary.read_text())
    validation = calibration["validation_train_argmax_current_phoenix"]
    if not validation["passes_validation_gate"]:
        raise RuntimeError("frozen validation candidate did not pass its gate")
    parameters = calibration["exploratory_train_argmax_parameters"]
    rows = load_jsonl(args.test_scores)
    baseline, calibrated = apply_parameters(rows, parameters)
    intent = load_intent_scores(args.intent_scores)
    phoenix_baseline = overlay_intent(rows, baseline, intent)
    phoenix_candidate = overlay_intent(rows, calibrated, intent)
    report = {
        "method": "q397_family_calibration_test_confirmation_v1",
        "frozen_parameters": parameters,
        "validation_reference": validation,
        "test_current_phoenix": summarize(
            rows, phoenix_baseline, phoenix_candidate
        ),
        "intent_rows": len(intent),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output_dir / "scores.jsonl").open("w") as handle:
        for row, base, candidate in zip(
            rows, phoenix_baseline, phoenix_candidate, strict=True
        ):
            handle.write(
                json.dumps(
                    {
                        "dataset": row["dataset"],
                        "family": row["family"],
                        "index": row["index"],
                        "label": row["label"],
                        "baseline": base,
                        "candidate": candidate,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-summary", type=Path, default=DEFAULT_CALIBRATION
    )
    parser.add_argument("--test-scores", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--intent-scores", type=Path, default=DEFAULT_INTENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
