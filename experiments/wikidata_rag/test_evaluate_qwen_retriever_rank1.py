import numpy as np

from experiments.wikidata_rag.evaluate_qwen_retriever_rank1 import (
    candidate_labels,
    filtered_plans,
    grounding_inputs,
    split_row_ids,
    subset_report,
)
from experiments.wikidata_rag.filter_qwen_planner_by_scores import score_map


def row(index: int, group: str = "group") -> dict:
    return {
        "dataset": "dataset",
        "index": index,
        "question_group": group,
        "question": "Who wrote it?",
        "answer_full": "Arthur wrote it.",
        "parse_error": False,
        "candidates": [
            {"id": "C00", "subject": "Book", "fact": "author: Wilkie"},
            {"id": "C01", "subject": "Book", "fact": "date: 1868"},
        ],
        "labels": [
            {"id": "C00", "label": "decisive"},
            {"id": "C01", "label": "relevant_insufficient"},
        ],
    }


def test_candidate_labels_follow_candidate_order() -> None:
    assert candidate_labels(row(1)) == [1, 0]


def test_subset_report_measures_top1_and_thresholded_retrieval() -> None:
    scores = np.asarray([0.9, 0.1, 0.2, 0.8])
    labels = np.asarray([1, 0, 0, 1])
    slices = [(0, 2), (2, 4)]

    report = subset_report(
        scores,
        labels,
        slices,
        [0, 1],
        absolute_threshold=0.5,
        margin_threshold=0.5,
    )

    assert report["candidate_auroc"] == 1.0
    assert report["top1_decisive_rows"] == 2
    assert report["absolute"]["precision"] == 1.0
    assert report["absolute"]["recall_of_decisive_rows"] == 1.0


def test_filtered_plans_preserve_grounded_metadata() -> None:
    plans = [{
        **row(1),
        "selected": [
            {"id": "C00", "claim_quote": "Arthur", "relation": "contradicts"},
            {"id": "C01", "claim_quote": "wrote", "relation": "supports"},
        ],
    }]
    scores = {
        ("dataset", 1, "C00"): 0.9,
        ("dataset", 1, "C01"): 0.2,
    }

    filtered = filtered_plans(plans, scores, 0.5)

    assert filtered[0]["selected"] == [plans[0]["selected"][0]]
    assert len(plans[0]["selected"]) == 2


def test_grounding_inputs_strip_labels_and_keep_only_top_candidate() -> None:
    rows = [row(1)]
    output = grounding_inputs(
        rows,
        np.asarray([0.8, 0.3]),
        [(0, 2)],
        0,
        0.5,
    )

    assert len(output) == 1
    assert output[0]["candidates"][0]["id"] == "C00"
    assert output[0]["retriever_score"] == 0.8
    assert "labels" not in output[0]


def test_split_row_ids_separates_novel_validation_groups() -> None:
    train = [row(1, "seen")]
    validation = [row(2, "seen"), row(3, "novel")]

    splits = split_row_ids(train, validation)

    assert splits["validation"] == [1, 2]
    assert splits["validation_seen"] == [1]
    assert splits["validation_novel"] == [2]


def test_score_map_uses_requested_frozen_score_field() -> None:
    scores = score_map([{
        "dataset": "dataset",
        "index": 1,
        "candidate_id": "C00",
        "base_score": 0.75,
        "adapter_score": 0.25,
    }], "base_score")

    assert scores == {("dataset", 1, "C00"): 0.75}
