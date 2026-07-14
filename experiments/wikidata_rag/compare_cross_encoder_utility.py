#!/usr/bin/env python3
"""Compare selected cross-encoder utility checkpoints without test selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def selected_checkpoint(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["selected"]
    for checkpoint in report["checkpoints"]:
        if checkpoint.get("stage") != "fine_tuned":
            continue
        if all(checkpoint.get(key) == selected.get(key) for key in (
            "target", "loss_mode", "query_mode", "score_mode",
            "frozen_bottom_layers",
            "learning_rate", "epoch",
        )):
            return checkpoint
    raise ValueError("Selected checkpoint is absent from report")


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "emitted": metrics["emitted"],
        "coverage": metrics["coverage"],
        "balanced_accuracy_delta": metrics["balanced_accuracy_delta"],
        "auroc_delta": metrics["auroc_delta"],
        "balanced_controlled_gain": metrics["balanced_controlled_gain"],
        "controlled_positive_precision": metrics["controlled_positive_precision"],
    }


def summarize(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    checkpoint = selected_checkpoint(report)
    zero = next(item for item in report["checkpoints"] if item["stage"] == "zero_shot")
    return {
        "model_name": report["model_name"],
        "selected": report["selected"],
        "model_bytes": report["selected_model_bytes"],
        "zero_shot": {
            split: compact_metrics(zero[split])
            for split in ("internal_test", "frozen_validation", "frozen_novel")
        },
        "fine_tuned": {
            split: compact_metrics(checkpoint[split])
            for split in ("calibration", "internal_test", "frozen_validation", "frozen_novel")
        },
        "candidate_quality": checkpoint["candidate_quality"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = [summarize(path) for path in args.reports]
    payload = {"models": summaries}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
