from __future__ import annotations

import math

import numpy as np

from experiments.wikidata_rag.analyze_counterfactual_utility import summarize
from experiments.wikidata_rag.build_utility_sweep_cache import deranged_donors
from experiments.wikidata_rag.score_counterfactual_utility import (
    correct_log_probability,
    rotated_controls,
)
from experiments.wikidata_rag.train_utility_retriever import (
    build_examples,
    selection_report,
    utility_feature_dict,
)


def candidate(identifier: str, utility: float = 0.2) -> dict:
    return {
        "id": identifier,
        "subject": "Paris",
        "fact": "country: France",
        "popularity": 10,
        "score": 0.8,
        "utility": utility,
        "controlled_utility": utility - 0.02,
        "semantic_label": "decisive",
    }


def row(number: int, label: int = 1) -> dict:
    return {
        "dataset": f"dev-varied-deception-d{number % 2}",
        "index": number,
        "label": label,
        "question": "Which country contains Paris?",
        "answer_full": "Paris is in France.",
        "rule_predicates": ["country"],
        "question_group": f"group-{number}",
        "empty_score": 0.4 if label else 0.2,
        "candidates": [candidate(f"C{number:02d}")],
        "shuffled_control": {"score": 0.42 if label else 0.22},
    }


def test_correct_log_probability_is_label_symmetric() -> None:
    assert math.isclose(
        correct_log_probability(0.8, 1), correct_log_probability(0.2, 0)
    )
    assert correct_log_probability(0.8, 1) > correct_log_probability(0.2, 1)


def test_rotated_controls_never_use_the_same_question_group() -> None:
    rows = [row(index) for index in range(4)]
    controls = rotated_controls(rows)

    assert all(control is not None for control in controls)
    assert [control["donor_index"] for control in controls] == [1, 2, 3, 0]


def test_reader_features_depend_on_the_frozen_student_margin() -> None:
    source = row(0)
    low = utility_feature_dict(
        source, source["candidates"][0], feature_mode="reader",
        semantic_score=0.0, semantic_rank=0,
    )
    source["empty_score"] = 0.9
    high = utility_feature_dict(
        source, source["candidates"][0], feature_mode="reader",
        semantic_score=0.0, semantic_rank=0,
    )

    assert low != high


def test_selection_report_implements_explicit_no_evidence_action() -> None:
    rows = [row(index, label=index % 2) for index in range(6)]
    predictions = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    slices = [(index, index + 1) for index in range(6)]

    report = selection_report(
        rows, predictions, slices, list(range(6)), threshold=0.0,
        minimum_gain=0.01,
    )

    assert report["emitted"] == 3
    assert report["controlled_positive"] == 3
    assert report["semantic_decisive_precision"] == 1.0


def test_deranged_donors_preserve_sparse_coverage() -> None:
    selected = [candidate("A"), None, candidate("B"), None, candidate("C")]
    donors = deranged_donors(selected)

    assert donors[1] is None and donors[3] is None
    assert [donors[index]["id"] for index in (0, 2, 4)] == ["B", "C", "A"]


def test_utility_summary_counts_candidate_rescues() -> None:
    source = row(0, label=1)
    source["empty_score"] = 0.4
    source["candidates"][0]["score"] = 0.8
    source["shuffled_utility"] = 0.02

    report = summarize([source])

    assert report["candidate_binary_changes"]["rescues"] == 1
    assert report["by_semantic_label"]["decisive"]["utility"]["count"] == 1


def test_binary_utility_target_records_rescue_over_empty_and_shuffle() -> None:
    source = row(0, label=1)
    source["empty_score"] = 0.4
    source["candidates"][0]["score"] = 0.8
    source["shuffled_control"]["score"] = 0.42

    _, targets, _ = build_examples([source], "generic", None)

    assert targets["binary_utility"].tolist() == [1.0]
    assert targets["controlled_binary_utility"].tolist() == [1.0]
