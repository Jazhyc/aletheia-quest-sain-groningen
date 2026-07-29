from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_ndif_packing.run import (
    build_packed_batch,
    packed_position_batches,
    workload_report,
)


def test_packed_position_batches_respect_both_caps() -> None:
    lengths = [100, 200, 300, 400, 500]
    batches = packed_position_batches(
        lengths,
        token_budget=700,
        max_sequences=3,
    )

    assert sorted(position for batch in batches for position in batch) == list(
        range(len(lengths))
    )
    assert all(
        sum(lengths[position] for position in batch) <= 700
        for batch in batches
    )
    assert all(len(batch) <= 3 for batch in batches)


def test_packed_batch_exposes_every_qwen35_boundary() -> None:
    packed = build_packed_batch(
        [[10, 11, 12], [20, 21], [30, 31, 32, 33]],
        [0, 1, 2],
    )

    assert packed["input_ids"].tolist() == [
        [10, 11, 12, 20, 21, 30, 31, 32, 33]
    ]
    assert packed["position_ids"].tolist() == [
        [0, 1, 2, 0, 1, 0, 1, 2, 3]
    ]
    assert packed["seq_idx"].tolist() == [
        [0, 0, 0, 1, 1, 2, 2, 2, 2]
    ]
    assert packed["cu_seq_lens_q"].tolist() == [0, 3, 5, 9]
    assert packed["cu_seq_lens_k"].tolist() == [0, 3, 5, 9]
    assert packed["logits_to_keep"].tolist() == [2, 4, 8]
    assert packed["max_length_q"] == 4
    assert packed["max_length_k"] == 4
    assert "attention_mask" not in packed
    assert "labels" not in packed


def test_workload_report_counts_removed_padding() -> None:
    padded = [
        (
            [0, 1],
            {
                "input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            },
        )
    ]
    packed = [
        (
            [0, 1],
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            },
        )
    ]
    report = workload_report(padded, packed, [[1, 2], [3, 4, 5]])

    assert report["input_tokens"] == 5
    assert report["padded_tokens"] == 6
    assert report["padding_tokens"] == 1
    assert report["packed_tokens"] == 5
    assert report["padded_traces"] == 1
    assert report["packed_traces"] == 1
