#!/usr/bin/env python3
"""Select a belief-grounded intent prompt under the frozen P68 train gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_blind_teachers import (
    load_teacher_frame,
)
from experiments.pid_specialist_ensemble.analyze_ensemble import (
    comparison_counts,
    metrics,
    parse_member,
)


MIN_BA_GAIN = 0.005
MAX_RECALL_LOSS = 0.01


def candidate_passes_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    comparison: dict[str, int],
) -> dict[str, bool]:
    """Return each preregistered P68 training-gate decision."""
    tolerance = 1e-12
    checks = {
        "balanced_accuracy_gain": (
            candidate["balanced_accuracy"] - baseline["balanced_accuracy"]
            >= MIN_BA_GAIN - tolerance
        ),
        "fpr_non_increase": candidate["fpr"] <= baseline["fpr"] + tolerance,
        "recall_preserved": (
            candidate["recall"] >= baseline["recall"] - MAX_RECALL_LOSS - tolerance
        ),
        "fix_break_ratio": comparison["fixes"] >= 2 * comparison["breaks"],
    }
    checks["passed"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    specs = [parse_member(value) for value in args.member]
    names = [name for name, _ in specs]
    if len(names) < 2 or names[0] != "baseline":
        raise ValueError("first member must be baseline followed by candidates")

    frame = load_teacher_frame(specs)
    labels = frame["label"].to_numpy(dtype=int)
    member_metrics = {
        name: metrics(frame, frame[name].to_numpy(dtype=float)) for name in names
    }
    comparisons = {
        name: comparison_counts(
            labels,
            frame["baseline"].to_numpy(dtype=float),
            frame[name].to_numpy(dtype=float),
        )
        for name in names[1:]
    }
    gates = {
        name: candidate_passes_gate(
            member_metrics["baseline"], member_metrics[name], comparisons[name]
        )
        for name in names[1:]
    }
    passing = [name for name in names[1:] if gates[name]["passed"]]
    selected = max(
        passing,
        key=lambda name: member_metrics[name]["balanced_accuracy"],
        default=None,
    )
    report = {
        "rows": len(frame),
        "members": names,
        "parse_errors": {
            name: int(frame[f"parse_error_{name}"].sum()) for name in names
        },
        "metrics": member_metrics,
        "versus_baseline": comparisons,
        "gate": {
            "min_balanced_accuracy_gain": MIN_BA_GAIN,
            "max_recall_loss": MAX_RECALL_LOSS,
            "candidate_checks": gates,
            "selected": selected,
        },
        "pairwise_disagreement": {
            f"{left}__{right}": int((frame[left] != frame[right]).sum())
            for position, left in enumerate(names)
            for right in names[position + 1 :]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
