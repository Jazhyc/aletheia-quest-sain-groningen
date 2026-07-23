#!/usr/bin/env python3
"""Verify the packaged Phoenix 3.0 deception-only decision rule."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_ensemble import (
    load_member_frame,
    metrics,
)


def main() -> None:
    frame = load_member_frame([
        (
            "deception",
            ROOT / "results/blackbox/qwen9b_pid_specialist_material_rank1_v1/"
            "test_heterogeneous_objectives_v1/generations.jsonl",
        ),
    ]).reset_index(drop=True)
    actual = metrics(frame, frame["deception"].to_numpy(dtype=float))
    expected = {
        "balanced_accuracy": 0.9071428571428571,
        "auroc": 0.9071428571428571,
        "recall": 0.8452380952380952,
        "fpr": 0.030952380952380953,
    }
    for key, value in expected.items():
        if abs(actual[key] - value) > 1e-12:
            raise AssertionError(f"{key}: {actual[key]} != {value}")
    print(json.dumps({
        "rows": len(frame),
        **expected,
        "replay": "exact",
    }, indent=2))


if __name__ == "__main__":
    main()
