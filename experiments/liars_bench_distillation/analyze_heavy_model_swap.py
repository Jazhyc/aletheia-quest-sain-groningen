#!/usr/bin/env python3
"""Apply the frozen external-preservation gate to the Qwen-27B judge swap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_swap_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_overall_ba_loss: float = 0.01,
    maximum_family_ba_loss: float = 0.03,
) -> dict[str, Any]:
    overall_delta = (
        candidate["baseline"]["balanced_accuracy"]
        - baseline["baseline"]["balanced_accuracy"]
    )
    family_deltas = {
        family: (
            candidate["baseline_by_family"][family]["balanced_accuracy"]
            - baseline["baseline_by_family"][family]["balanced_accuracy"]
        )
        for family in sorted(baseline["baseline_by_family"])
    }
    return {
        "maximum_overall_ba_loss": maximum_overall_ba_loss,
        "maximum_family_ba_loss": maximum_family_ba_loss,
        "overall_ba_delta": overall_delta,
        "family_ba_deltas": family_deltas,
        "passed": bool(
            overall_delta >= -maximum_overall_ba_loss
            and min(family_deltas.values(), default=0.0) >= -maximum_family_ba_loss
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = {
        "baseline": args.baseline.as_posix(),
        "candidate": args.candidate.as_posix(),
        "gate": model_swap_gate(baseline, candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
