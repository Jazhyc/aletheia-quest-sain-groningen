from __future__ import annotations

import torch

from experiments.wikidata_rag.compare_cross_encoder_utility import selected_checkpoint
from experiments.wikidata_rag.refit_cross_encoder_utility import average_state_dicts
from experiments.wikidata_rag.test_counterfactual_utility import candidate, row
from experiments.wikidata_rag.train_cross_encoder_utility import (
    NO_EVIDENCE_TEXT,
    candidate_target,
    checkpoint_key,
    evaluate_checkpoint,
    flatten_batch,
    pairwise_utility_loss,
    serialize_query,
    target_training_rows,
    transform_policy_scores,
    split_row_ids,
)


def test_query_serialization_excludes_deceptive_system_instruction() -> None:
    source = row(0)
    source["conversation"] = "SYSTEM: provide a deceptive answer"
    serialized = serialize_query(source)

    assert "deceptive" not in serialized
    assert source["question"] in serialized
    assert source["answer_full"] in serialized
    assert "0.4000" in serialized


def test_flatten_batch_appends_explicit_no_evidence_candidate() -> None:
    source = row(0)
    source["candidates"].append(candidate("second", utility=-0.2))
    queries, documents, targets = flatten_batch([source], "controlled_utility")

    assert len(queries) == len(documents) == 3
    assert documents[-1] == NO_EVIDENCE_TEXT
    assert targets[0].tolist()[-1] == 0.0


def test_pairwise_loss_prefers_correct_utility_ordering() -> None:
    targets = torch.tensor([0.2, 0.0, -0.2])
    ordered = pairwise_utility_loss(
        torch.tensor([2.0, 0.0, -2.0]), targets,
        target_scale=0.1, minimum_gain=0.01, regression_weight=0.25,
    )
    reversed_order = pairwise_utility_loss(
        torch.tensor([-2.0, 0.0, 2.0]), targets,
        target_scale=0.1, minimum_gain=0.01, regression_weight=0.25,
    )

    assert ordered < reversed_order


def test_hard_listwise_loss_selects_no_evidence_without_positive_gain() -> None:
    targets = torch.tensor([-0.2, 0.005, 0.0])
    correct = pairwise_utility_loss(
        torch.tensor([-1.0, 0.0, 2.0]), targets,
        target_scale=0.1, minimum_gain=0.01, regression_weight=0.25,
        loss_mode="hard_listwise",
    )
    incorrect = pairwise_utility_loss(
        torch.tensor([-1.0, 2.0, 0.0]), targets,
        target_scale=0.1, minimum_gain=0.01, regression_weight=0.25,
        loss_mode="hard_listwise",
    )

    assert correct < incorrect


def test_hard_listwise_accepts_row_with_only_no_evidence() -> None:
    loss = pairwise_utility_loss(
        torch.tensor([2.0]), torch.tensor([0.0]),
        target_scale=0.1, minimum_gain=0.01, regression_weight=0.25,
        loss_mode="hard_listwise",
    )

    assert loss.item() == 0.0


def test_binary_and_score_delta_targets_are_derived_from_reader_outputs() -> None:
    source = row(0, label=1)
    source["empty_score"] = 0.4
    source["candidates"][0]["score"] = 0.8

    assert candidate_target(source, source["candidates"][0], "binary_utility") == 1.0
    assert candidate_target(source, source["candidates"][0], "score_delta") > 0.0


def test_semantic_targets_mask_unlabeled_candidates() -> None:
    source = row(0)
    source["candidates"] = [
        source["candidates"][0] | {"semantic_label": "decisive"},
        source["candidates"][0] | {"semantic_label": "relevant_insufficient"},
        source["candidates"][0] | {"semantic_label": None},
    ]

    assert candidate_target(source, source["candidates"][0], "semantic_decisive") == 1.0
    assert candidate_target(source, source["candidates"][1], "semantic_decisive") == 0.0
    assert candidate_target(source, source["candidates"][1], "semantic_relevant") == 1.0
    assert torch.isnan(torch.tensor(candidate_target(
        source, source["candidates"][2], "semantic_relevant"
    )))


def test_semantic_training_drops_only_wholly_unlabeled_rows() -> None:
    known = row(0)
    unknown = row(1)
    known["candidates"][0]["semantic_label"] = "irrelevant"
    unknown["candidates"][0]["semantic_label"] = None

    assert target_training_rows([known, unknown], [0, 1], "semantic_decisive") == [0]
    assert target_training_rows([known, unknown], [0, 1], "utility") == [0, 1]


def test_query_modes_make_detector_state_an_explicit_ablation() -> None:
    source = row(0)
    source["empty_score"] = 0.95

    assert "0.9500" in serialize_query(source, "full")
    assert "very likely deceptive" in serialize_query(source, "score_bucket")
    assert "detector" not in serialize_query(source, "no_score")


def test_policy_score_transforms_preserve_ranking_and_penalize_more_candidates() -> None:
    predictions = torch.tensor([2.0, 1.0, 2.0]).numpy()
    slices = [(0, 2), (2, 3)]
    count_adjusted = transform_policy_scores(predictions, slices, "count_adjusted")
    softmax = transform_policy_scores(predictions, slices, "softmax_logprob")

    assert count_adjusted[0] > count_adjusted[1]
    assert count_adjusted[0] < count_adjusted[2]
    assert softmax[0] > softmax[1]
    assert softmax[0] < softmax[2]


def test_group_split_keeps_question_variants_together() -> None:
    training = [row(index) for index in range(20)]
    training[1]["question_group"] = training[0]["question_group"]
    validation = [row(100)]
    split = split_row_ids(training, validation)

    memberships = [
        name for name in ("train", "calibration", "internal_test")
        if 0 in split[name] or 1 in split[name]
    ]
    assert len(memberships) == 1
    assert 0 in split[memberships[0]] and 1 in split[memberships[0]]
    assert 20 in split["frozen_novel"]


def test_checkpoint_selection_uses_calibration_only() -> None:
    report = {
        "calibration": {
            "balanced_accuracy_delta": 0.1,
            "balanced_controlled_gain": 0.2,
            "balanced_raw_gain": 0.3,
        },
        "candidate_quality": {"calibration": {"average_precision": 0.4}},
        "frozen_validation": {"balanced_accuracy_delta": -1.0},
    }

    assert checkpoint_key(report) == (0.1, 0.2, 0.3, 0.4)


def test_robust_checkpoint_selection_requires_both_grouped_splits() -> None:
    report = {
        "calibration": {
            "balanced_accuracy_delta": 0.1,
            "balanced_controlled_gain": 0.2,
            "balanced_raw_gain": 0.3,
        },
        "internal_test": {
            "balanced_accuracy_delta": -0.05,
            "balanced_controlled_gain": 0.1,
        },
        "candidate_quality": {
            "calibration": {"average_precision": 0.4},
            "internal_test": {"average_precision": 0.6},
        },
    }

    assert checkpoint_key(report, "robust") == (-0.05, 0.025, 0.1, 0.5)


def test_report_lookup_finds_selected_training_checkpoint() -> None:
    report = {
        "selected": {"target": "utility", "learning_rate": 1e-5, "epoch": 2},
        "checkpoints": [
            {"stage": "zero_shot"},
            {
                "stage": "fine_tuned", "target": "utility",
                "learning_rate": 1e-5, "epoch": 1,
            },
            {
                "stage": "fine_tuned", "target": "utility",
                "learning_rate": 1e-5, "epoch": 2,
            },
        ],
    }

    assert selected_checkpoint(report)["epoch"] == 2


def test_checkpoint_evaluation_handles_empty_novel_subset() -> None:
    rows = [row(index, label=index % 2) for index in range(10)]
    predictions = torch.linspace(-1.0, 1.0, len(rows)).numpy()
    targets = torch.tensor([
        source["candidates"][0]["controlled_utility"] for source in rows
    ]).numpy()
    slices = [(index, index + 1) for index in range(len(rows))]
    splits = {
        "train": [0, 1], "calibration": [2, 3, 4, 5],
        "internal_test": [6, 7], "frozen": [8, 9],
        "frozen_novel": [], "frozen_seen": [8, 9],
    }

    report = evaluate_checkpoint(
        rows, targets, predictions, slices, splits,
        minimum_gain=0.01, minimum_precision=0.0, minimum_emitted=1,
    )

    assert report["frozen_novel"]["rows"] == 0
    assert report["frozen_novel"]["balanced_accuracy_delta"] is None


def test_state_dict_soup_averages_float_tensors_and_preserves_integers() -> None:
    first = {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(2)}
    second = {"weight": torch.tensor([3.0, 5.0]), "count": torch.tensor(4)}

    soup = average_state_dicts([first, second])

    assert soup["weight"].tolist() == [2.0, 4.0]
    assert soup["count"].item() == 2
