from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.analyze_transfer import (
    accepts_action_route,
    paired_changes,
)


def write_rows(path: Path, predictions: list[int | None]) -> None:
    rows = [
        {
            "dataset": "unit",
            "index": str(index),
            "label": label,
            "prediction": prediction,
        }
        for index, (label, prediction) in enumerate(zip((0, 1, 1), predictions))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_paired_changes_counts_fixes_and_breaks(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_rows(baseline, [0, 0, 1])
    write_rows(candidate, [1, 1, 1])

    assert paired_changes(baseline, candidate) == {
        "changes": 2,
        "fixes": 1,
        "breaks": 1,
    }


def test_paired_changes_treats_parse_failure_as_negative(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_rows(baseline, [0, None, 1])
    write_rows(candidate, [0, 1, 1])

    assert paired_changes(baseline, candidate) == {
        "changes": 1,
        "fixes": 1,
        "breaks": 0,
    }


def test_action_route_requires_large_target_gain_and_zero_spillover() -> None:
    delta = {"category_deltas": {"insider-trading": 0.12}}

    assert accepts_action_route(
        delta,
        {"per_category": {"insider-trading": 200}},
    )
    assert not accepts_action_route(
        delta,
        {"per_category": {"insider-trading": 200, "soft-trigger": 1}},
    )
    assert not accepts_action_route(
        {"category_deltas": {"insider-trading": 0.09}},
        {"per_category": {"insider-trading": 200}},
    )
