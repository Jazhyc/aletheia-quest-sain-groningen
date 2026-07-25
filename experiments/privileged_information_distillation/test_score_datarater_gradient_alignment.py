import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.score_datarater_gradient_alignment import (
    gradient_alignment,
    limit_records_balanced,
    per_sequence_completion_loss,
    record_key,
    select_random_fraction,
    select_scored_fraction,
    split_meta_records,
    write_jsonl,
)


def records() -> list[dict[str, object]]:
    return [
        {
            "dataset": f"dataset-{dataset}",
            "index": 100 * label + index,
            "label": label,
        }
        for dataset in ("a", "b")
        for label in (0, 1)
        for index in range(10)
    ]


def test_split_meta_records_is_balanced_disjoint_and_order_stable() -> None:
    source = records()
    meta, candidates = split_meta_records(source, 0.2, seed=3)
    reversed_meta, _ = split_meta_records(list(reversed(source)), 0.2, seed=3)

    assert len(meta) == 8
    assert len(candidates) == 32
    assert {record_key(record) for record in meta}.isdisjoint(
        record_key(record) for record in candidates
    )
    assert {record_key(record) for record in meta} == {
        record_key(record) for record in reversed_meta
    }
    assert {
        (record["dataset"], record["label"]): sum(
            candidate["dataset"] == record["dataset"]
            and candidate["label"] == record["label"]
            for candidate in meta
        )
        for record in meta
    } == {
        ("dataset-a", 0): 2,
        ("dataset-a", 1): 2,
        ("dataset-b", 0): 2,
        ("dataset-b", 1): 2,
    }


def test_scored_and_random_selection_preserve_every_stratum() -> None:
    source = records()
    scores = {record_key(record): float(record["index"]) for record in source}

    selected = select_scored_fraction(source, scores, 0.3, seed=0)
    random_selected = select_random_fraction(source, 0.3, seed=0)

    assert len(selected) == len(random_selected) == 12
    for dataset in ("dataset-a", "dataset-b"):
        for label in (0, 1):
            selected_indices = [
                int(record["index"]) % 100
                for record in selected
                if record["dataset"] == dataset and record["label"] == label
            ]
            assert selected_indices == [7, 8, 9]
            assert sum(
                record["dataset"] == dataset and record["label"] == label
                for record in random_selected
            ) == 3


def test_limit_records_balanced_round_robins_strata() -> None:
    selected = limit_records_balanced(records(), 6, seed=1, namespace="test")

    counts: dict[tuple[object, object], int] = {}
    for record in selected:
        key = record["dataset"], record["label"]
        counts[key] = counts.get(key, 0) + 1
    assert sorted(counts.values()) == [1, 1, 2, 2]


def test_per_sequence_completion_loss_ignores_prompt_and_averages_by_row() -> None:
    labels = torch.tensor([[-100, 1, 0], [-100, -100, 1]])
    logits = torch.tensor(
        [
            [[0.0, 2.0], [2.0, 0.0], [9.0, -9.0]],
            [[0.0, 0.0], [0.0, 3.0], [9.0, -9.0]],
        ]
    )

    losses = per_sequence_completion_loss(logits, labels)

    expected_first = (
        torch.nn.functional.cross_entropy(logits[0, 0:1], labels[0, 1:2])
        + torch.nn.functional.cross_entropy(logits[0, 1:2], labels[0, 2:3])
    ) / 2
    expected_second = torch.nn.functional.cross_entropy(
        logits[1, 1:2], labels[1, 2:3]
    )
    assert torch.allclose(losses, torch.stack([expected_first, expected_second]))


def test_gradient_alignment_returns_dot_cosine_and_norm() -> None:
    gradients = [torch.tensor([1.0, 2.0])]
    reference = [torch.tensor([2.0, 0.0])]

    dot, cosine, norm = gradient_alignment(gradients, reference, reference_norm=2.0)

    assert dot == 2.0
    assert abs(cosine - 1 / 5**0.5) < 1e-7
    assert abs(norm - 5**0.5) < 1e-7


def test_write_jsonl_creates_strict_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, [{"dataset": "d", "index": 1, "label": 0}])

    assert json.loads(path.read_text()) == {
        "dataset": "d",
        "index": 1,
        "label": 0,
    }
