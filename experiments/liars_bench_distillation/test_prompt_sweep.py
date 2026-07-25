from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.analyze_prompt_validation import (
    analyze_validation,
)
from experiments.liars_bench_distillation.evaluate_prompt_sweep import split_records
from experiments.liars_bench_distillation.select_prompt_sweep import select_prompt


def test_prompt_split_is_balanced_disjoint_and_deterministic() -> None:
    records = [
        {
            "dataset": f"liars-bench/{category}",
            "index": f"{category}:{label}:{index}",
            "category": category,
            "label": label,
        }
        for category in ("choice", "report")
        for label in (0, 1)
        for index in range(10)
    ]

    development = split_records(records, "development", seed=7)
    confirmation = split_records(records, "confirmation", seed=7)

    assert len(development) == len(confirmation) == 20
    assert {
        (row["dataset"], row["index"]) for row in development
    }.isdisjoint({
        (row["dataset"], row["index"]) for row in confirmation
    })
    assert split_records(records, "development", seed=7) == development
    assert {
        (category, label): sum(
            row["category"] == category and row["label"] == label
            for row in development
        )
        for category in ("choice", "report")
        for label in (0, 1)
    } == {
        ("choice", 0): 5,
        ("choice", 1): 5,
        ("report", 0): 5,
        ("report", 1): 5,
    }


def condition(
    macro: float,
    *,
    choice: float,
    report: float,
    choice_cell: float,
    report_cell: float,
    parse_errors: int = 0,
) -> dict:
    return {
        "macro_category_balanced_accuracy": macro,
        "metrics": {"balanced_accuracy": macro},
        "per_category": {
            "choice": {"balanced_accuracy": choice, "accuracy": choice},
            "report": {"balanced_accuracy": report, "accuracy": report},
        },
        "per_source_model": {
            "model": {"balanced_accuracy": macro, "accuracy": macro},
        },
        "per_category_source_model": {
            "choice::model": {
                "balanced_accuracy": choice_cell,
                "accuracy": choice_cell,
            },
            "report::model": {
                "balanced_accuracy": report_cell,
                "accuracy": report_cell,
            },
        },
        "parse_errors": parse_errors,
    }


def test_selection_requires_gain_and_group_preservation() -> None:
    result = {
        "split": "development",
        "conditions": {
            "control": condition(
                0.60,
                choice=0.60,
                report=0.60,
                choice_cell=0.60,
                report_cell=0.60,
            ),
            "good": condition(
                0.65,
                choice=0.64,
                report=0.66,
                choice_cell=0.64,
                report_cell=0.66,
            ),
            "hidden_regression": condition(
                0.66,
                choice=0.72,
                report=0.60,
                choice_cell=0.72,
                report_cell=0.54,
            ),
        },
    }

    report = select_prompt(
        result,
        baseline_name="control",
        minimum_macro_gain=0.03,
        minimum_category_delta=-0.02,
        minimum_category_source_model_delta=-0.05,
        maximum_parse_error_increase=10,
    )

    assert report["selected"] == "good"
    assert report["comparisons"]["good"]["passes"]
    assert not report["comparisons"]["hidden_regression"]["passes"]


def validation_result(
    all_ba: float,
    instructed_ba: float,
    varied_ba: float,
    *,
    parse_errors: int,
) -> dict:
    return {
        "metrics": {
            "all": {"balanced_accuracy": all_ba},
            "instructed": {"balanced_accuracy": instructed_ba},
            "varied": {"balanced_accuracy": varied_ba},
        },
        "parse_errors": parse_errors,
    }


def test_validation_requires_external_confirmation_and_preservation() -> None:
    control = validation_result(0.90, 0.95, 0.82, parse_errors=2)
    candidate = validation_result(0.899, 0.945, 0.815, parse_errors=3)

    accepted = analyze_validation(
        control,
        candidate,
        {"selected": "mode_first"},
        candidate_name="mode_first",
        maximum_balanced_accuracy_loss=0.0025,
        maximum_scenario_loss=0.01,
        maximum_parse_error_increase=10,
    )
    rejected = analyze_validation(
        control,
        candidate,
        {"selected": None},
        candidate_name="mode_first",
        maximum_balanced_accuracy_loss=0.0025,
        maximum_scenario_loss=0.01,
        maximum_parse_error_increase=10,
    )

    assert accepted["accepted"]
    assert not rejected["accepted"]
