import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CALIBRATION_LAUNCHER = (
    ROOT
    / "experiments/privileged_information_distillation"
    / "submit_qwen397_soft_datarater_calibration.sh"
)

from experiments.privileged_information_distillation.score_datarater_gradient_alignment import (
    finite_difference_alignment,
    gradient_alignment,
    limit_records_balanced,
    load_scored_rows,
    per_sequence_direct_soft_binary_loss,
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


def test_direct_soft_binary_loss_scores_last_unpadded_boundary() -> None:
    logits = torch.zeros((2, 3, 4))
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    target_ids = torch.tensor([0, 2])
    soft_targets = torch.tensor([0.25, 0.75])
    logits[0, 1, 0] = 1.0
    logits[0, 1, 2] = -1.0
    logits[1, 2, 0] = -1.5
    logits[1, 2, 2] = 1.5
    logits[0, 2, 2] = 100.0

    losses = per_sequence_direct_soft_binary_loss(
        logits,
        attention_mask,
        soft_targets,
        target_ids,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([-2.0, 3.0]),
        soft_targets,
        reduction="none",
    )

    assert torch.allclose(losses, expected)


def test_gradient_alignment_returns_dot_cosine_and_norm() -> None:
    gradients = [torch.tensor([1.0, 2.0])]
    reference = [torch.tensor([2.0, 0.0])]

    dot, cosine, norm = gradient_alignment(gradients, reference, reference_norm=2.0)

    assert dot == 2.0
    assert abs(cosine - 1 / 5**0.5) < 1e-7
    assert abs(norm - 5**0.5) < 1e-7


def test_finite_difference_alignment_matches_exact_directional_derivative() -> None:
    class TinyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.1]))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> SimpleNamespace:
            del attention_mask
            logits = self.weight.view(1, 1, 2).expand(
                input_ids.shape[0],
                input_ids.shape[1],
                2,
            )
            return SimpleNamespace(logits=logits)

    model = TinyLM()
    original_weight = model.weight.detach().clone()
    input_ids = torch.zeros((2, 3), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, 0, 1], [-100, 1, 1]])
    reference = [torch.tensor([0.3, -0.4])]
    reference_norm = 0.5
    losses, approximate_dots = finite_difference_alignment(
        model,
        [model.weight],
        reference,
        reference_norm,
        (input_ids, attention_mask, labels),
        epsilon=1e-3,
    )

    exact_dots = []
    sequence_losses = per_sequence_completion_loss(
        model(input_ids, attention_mask).logits,
        labels,
    )
    for position in range(2):
        gradient = torch.autograd.grad(
            sequence_losses[position],
            model.weight,
            retain_graph=True,
        )[0]
        exact_dots.append(float(torch.sum(gradient * reference[0])))

    assert len(losses) == 2
    assert torch.allclose(
        torch.tensor(approximate_dots),
        torch.tensor(exact_dots),
        atol=2e-4,
        rtol=2e-3,
    )
    assert torch.equal(model.weight, original_weight)


def test_finite_difference_alignment_restores_low_precision_parameters() -> None:
    class TinyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([0.203125, -0.1015625], dtype=torch.bfloat16)
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> SimpleNamespace:
            del attention_mask
            logits = self.weight.float().view(1, 1, 2).expand(
                input_ids.shape[0],
                input_ids.shape[1],
                2,
            )
            return SimpleNamespace(logits=logits)

    model = TinyLM()
    original_weight = model.weight.detach().clone()
    finite_difference_alignment(
        model,
        [model.weight],
        [torch.tensor([0.3, -0.4])],
        0.5,
        (
            torch.zeros((1, 3), dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
            torch.tensor([[-100, 0, 1]]),
        ),
        epsilon=0.1,
    )

    assert torch.equal(model.weight, original_weight)


def test_write_jsonl_creates_strict_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, [{"dataset": "d", "index": 1, "label": 0}])

    assert json.loads(path.read_text()) == {
        "dataset": "d",
        "index": 1,
        "label": 0,
    }


def test_load_scored_rows_supports_resume_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.jsonl"
    row = {
        "dataset": "d",
        "index": 1,
        "label": 0,
        "gradient_cosine": 0.2,
    }
    write_jsonl(path, [row])

    assert load_scored_rows(path) == {("d", 1): row}

    write_jsonl(path, [row, row])
    try:
        load_scored_rows(path)
    except ValueError as error:
        assert "duplicate score row" in str(error)
    else:
        raise AssertionError("duplicate score checkpoints must fail")


def test_qwen397_soft_calibration_compares_exact_and_three_fd_steps() -> None:
    source = CALIBRATION_LAUNCHER.read_text()

    assert 'sha256sum -c "${CACHE_ROOT}/SHA256SUMS"' in source
    assert "--objective soft_binary" in source
    assert '--soft-teacher-artifact "${CACHE_ROOT}/soft_targets.jsonl"' in source
    assert "--scoring-mode exact" in source
    for epsilon in ("0.01", "0.03", "0.1"):
        assert f"--finite-difference-epsilon {epsilon}" in source
    assert "--max-meta-records 36" in source
    assert "--max-candidates 72" in source
