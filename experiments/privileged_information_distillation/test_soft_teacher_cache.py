from __future__ import annotations

import json

import pytest
import torch

from experiments.privileged_information_distillation.build_soft_teacher_cache import (
    aggregate_soft_targets,
)
from experiments.privileged_information_distillation.train_student_sft import (
    attach_soft_teacher_targets,
    soft_binary_distillation_loss,
    soft_rating_distillation_loss,
)


def member(dataset: str, index: int, name: str, score: float, label: int = 1):
    return {
        "dataset": dataset,
        "index": index,
        "label": label,
        "ensemble_member": name,
        "score": score,
        "rating_probs": {"1": 0.5, "7": 0.5},
        "missing_rating_token_ids": [],
        "parse_error": False,
    }


def test_aggregate_soft_targets_selects_max_and_normalizes_without_labels() -> None:
    records = [
        member("d", 1, "a", 0.49),
        member("d", 1, "b", 0.50),
        member("d", 1, "c", 0.51),
        member("d", 2, "a", 0.48, label=0),
        member("d", 2, "b", 0.49, label=0),
        member("d", 2, "c", 0.50, label=0),
    ]

    aggregated = aggregate_soft_targets(records)

    assert [record["selected_member"] for record in aggregated] == ["c", "c"]
    assert aggregated[0]["soft_target"] > aggregated[1]["soft_target"]
    assert aggregated[0]["normalization"] == aggregated[1]["normalization"]


def test_aggregate_soft_targets_rejects_incomplete_members() -> None:
    with pytest.raises(ValueError, match="expected 3 teacher members"):
        aggregate_soft_targets([member("d", 1, "a", 0.5)])


def test_attach_and_soft_binary_loss(tmp_path) -> None:
    artifact = tmp_path / "soft.jsonl"
    artifact.write_text(json.dumps({
        "dataset": "d",
        "index": 1,
        "label": 1,
        "soft_target": 0.75,
    }) + "\n")
    attached = attach_soft_teacher_targets(
        [{"dataset": "d", "index": 1, "label": 1}],
        artifact,
    )
    logits = torch.tensor([[0.0, 1.0]])
    loss = soft_binary_distillation_loss(
        logits,
        torch.tensor([attached[0]["_soft_target"]]),
    )

    assert loss.item() > 0


def test_soft_rating_loss_matches_normalized_teacher_distribution() -> None:
    targets = torch.tensor([
        [0.02, 0.03, 0.05, 0.10, 0.15, 0.25, 0.40],
        [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02],
    ])
    matching_logits = targets.log()
    mismatched_logits = matching_logits.flip(dims=(-1,))

    matching_loss = soft_rating_distillation_loss(matching_logits, targets)
    mismatched_loss = soft_rating_distillation_loss(mismatched_logits, targets)

    assert matching_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert mismatched_loss > matching_loss


def test_soft_rating_loss_rejects_unnormalized_targets() -> None:
    with pytest.raises(ValueError, match="normalize"):
        soft_rating_distillation_loss(
            torch.zeros((1, 7)),
            torch.ones((1, 7)),
        )
