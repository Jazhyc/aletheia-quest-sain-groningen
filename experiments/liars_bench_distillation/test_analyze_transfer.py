from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.analyze_transfer import (
    accepts_action_route,
    external_delta,
    paired_changes,
    paired_changes_grouped,
    passes_external_floors,
)
from experiments.liars_bench_distillation.evaluate_students import balanced_accuracy


def write_rows(path: Path, predictions: list[int | None]) -> None:
    rows = [
        {
            "dataset": "unit",
            "index": str(index),
            "label": label,
            "prediction": prediction,
            "category": "first" if index < 2 else "second",
            "source_model": "model-a" if index != 1 else "model-b",
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


def test_grouped_external_changes_preserve_fix_break_counts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_rows(baseline, [0, 0, 1])
    write_rows(candidate, [1, 1, 1])

    grouped = paired_changes_grouped(baseline, candidate)

    assert grouped["all"] == {"changes": 2, "fixes": 1, "breaks": 1}
    assert grouped["by_category"] == {
        "first": {"changes": 2, "fixes": 1, "breaks": 1}
    }
    assert grouped["by_source_model"] == {
        "model-a": {"changes": 1, "fixes": 0, "breaks": 1},
        "model-b": {"changes": 1, "fixes": 1, "breaks": 0},
    }


def test_external_floors_reject_hidden_group_regression() -> None:
    delta = {
        "category_deltas": {"choice": 0.05, "action": -0.021},
        "source_model_deltas": {"qwen": 0.03, "mistral": -0.04},
        "category_source_model_deltas": {
            "choice::qwen": 0.03,
            "action::mistral": -0.051,
        },
    }

    assert not passes_external_floors(
        delta,
        minimum_category_delta=-0.02,
        minimum_source_model_delta=-0.05,
        minimum_category_source_model_delta=-0.05,
    )
    assert passes_external_floors(
        delta,
        minimum_category_delta=-0.03,
        minimum_source_model_delta=-0.05,
        minimum_category_source_model_delta=-0.06,
    )


def test_one_class_group_reports_accuracy_and_undefined_balanced_accuracy() -> None:
    metrics = balanced_accuracy([
        {"label": 1, "prediction": 1},
        {"label": 1, "prediction": 0},
    ])

    assert metrics == {
        "balanced_accuracy": None,
        "accuracy": 0.5,
        "recall": 0.5,
        "fpr": None,
        "positive_rows": 2,
        "negative_rows": 0,
    }


def test_external_delta_uses_accuracy_for_one_class_source() -> None:
    baseline = {
        "macro_category_balanced_accuracy": 0.5,
        "per_category": {"action": {"balanced_accuracy": 0.5}},
        "per_source_model": {
            "kimi": {"balanced_accuracy": None, "accuracy": 0.75},
        },
        "per_category_source_model": {},
    }
    candidate = {
        "macro_category_balanced_accuracy": 0.6,
        "per_category": {"action": {"balanced_accuracy": 0.6}},
        "per_source_model": {
            "kimi": {"balanced_accuracy": None, "accuracy": 1.0},
        },
        "per_category_source_model": {},
    }

    delta = external_delta(candidate, baseline)

    assert delta["source_model_deltas"] == {"kimi": 0.25}
    assert delta["source_model_metrics"] == {"kimi": "accuracy"}
