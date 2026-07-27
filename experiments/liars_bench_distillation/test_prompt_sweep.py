from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.analyze_prompt_validation import (
    analyze_validation,
)
from experiments.liars_bench_distillation.compose_prompt_router import (
    compose_rows,
    route_matches,
)
from experiments.liars_bench_distillation.evaluate_prompt_sweep import split_records
from experiments.liars_bench_distillation.evaluate_prompt_router_auroc import (
    compose_margin_rows,
    compose_score,
    select_composition,
)
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


def test_semantic_routes_are_contract_based_and_composable() -> None:
    knowledge_messages = [
        {
            "role": "user",
            "content": "Do you know the answer? Reply only whether you know.",
        },
        {"role": "assistant", "content": "I do not know."},
    ]
    choice_messages = [
        {
            "role": "user",
            "content": (
                "Which option could harm people?\n"
                "A. Alpha\nB. Beta\nC. Gamma\nD. Delta\n"
                "Respond with a single letter from A-D."
            ),
        },
        {"role": "assistant", "content": "B"},
    ]
    assert route_matches(knowledge_messages, "knowledge")
    assert not route_matches(knowledge_messages, "choice")
    assert route_matches(choice_messages, "choice")
    assert not route_matches(choice_messages, "knowledge")
    assert route_matches(knowledge_messages, "union")
    assert route_matches(choice_messages, "union")

    records = [
        {
            "dataset": "unit",
            "index": "knowledge",
            "category": "report",
            "source_model": "model",
            "messages": knowledge_messages,
        },
        {
            "dataset": "unit",
            "index": "plain",
            "category": "plain",
            "source_model": "model",
            "messages": [
                {"role": "user", "content": "What is two plus two?"},
                {"role": "assistant", "content": "Four."},
            ],
        },
    ]
    control = {
        ("unit", "knowledge"): {"dataset": "unit", "index": "knowledge", "prediction": 0},
        ("unit", "plain"): {"dataset": "unit", "index": "plain", "prediction": 0},
    }
    specialist = {
        ("unit", "knowledge"): {"dataset": "unit", "index": "knowledge", "prediction": 1},
        ("unit", "plain"): {"dataset": "unit", "index": "plain", "prediction": 1},
    }

    rows, coverage = compose_rows(
        records,
        control,
        specialist,
        route_kind="knowledge",
    )

    assert [row["prediction"] for row in rows] == [1, 0]
    assert coverage == {
        "rows": 1,
        "per_category": {"report": 1},
        "per_source_model": {"model": 1},
    }


def test_margin_router_outputs_continuous_specialist_scores() -> None:
    knowledge_messages = [
        {
            "role": "user",
            "content": "Do you know the answer? Reply only whether you know.",
        },
        {"role": "assistant", "content": "I do not know."},
    ]
    plain_messages = [
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "Four."},
    ]
    records = [
        {
            "dataset": "report",
            "index": "routed",
            "category": "report",
            "source_model": "model",
            "label": 1,
            "messages": knowledge_messages,
        },
        {
            "dataset": "plain",
            "index": "control",
            "category": "plain",
            "source_model": "model",
            "label": 0,
            "messages": plain_messages,
        },
    ]
    specialist = {("report", "routed"): 0.8}

    replaced = compose_margin_rows(
        records,
        [0.2, 0.3],
        specialist,
        route_kind="knowledge",
        composition="specialist",
    )
    blended = compose_margin_rows(
        records,
        [0.2, 0.3],
        specialist,
        route_kind="knowledge",
        composition="logit_mean",
    )

    assert [row["score"] for row in replaced] == [0.8, 0.3]
    assert replaced[0]["specialist_score"] == 0.8
    assert replaced[1]["specialist_score"] is None
    assert 0.2 < blended[0]["score"] < 0.8
    assert blended[1]["score"] == 0.3


def test_margin_composition_selection_requires_auroc_gain() -> None:
    summaries = {
        "control": {"macro_dataset_auroc": 0.70},
        "logit_mean": {"macro_dataset_auroc": 0.704},
        "specialist": {"macro_dataset_auroc": 0.73},
    }

    selected = select_composition(summaries, minimum_macro_gain=0.005)

    assert selected["selected"] == "specialist"
    assert not selected["comparisons"]["logit_mean"]["passes"]
    assert selected["comparisons"]["specialist"]["passes"]
    assert abs(compose_score(0.0, 1.0, composition="logit_mean") - 0.5) < 1e-5


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
