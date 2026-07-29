from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_ndif_batch_width.run import (
    encode_batches,
    position_batches,
    workload_report,
)
from experiments.phoenix_ndif_batch_width.run_dynamic import (
    dynamic_position_batches,
)


class FakeTokenizer:
    def pad(self, rows, *, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        maximum = max(len(row["input_ids"]) for row in rows)
        ids = []
        masks = []
        for row in rows:
            values = list(row["input_ids"])
            missing = maximum - len(values)
            ids.append([0] * missing + values)
            masks.append([0] * missing + [1] * len(values))
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


def test_short_width_changes_only_short_tier() -> None:
    lengths = [500] * 56 + [700] * 32 + [1_000] * 16

    batches_48 = position_batches(lengths, short_width=48)
    batches_56 = position_batches(lengths, short_width=56)

    assert [len(batch) for batch in batches_48] == [48, 32, 16, 8]
    assert [len(batch) for batch in batches_56] == [56, 32, 16]
    assert max(lengths[position] for position in batches_56[-1]) == 1_000


def test_encode_batches_left_pads_without_changing_tokens() -> None:
    tokenizer = FakeTokenizer()
    token_ids = [[1, 2], [3, 4, 5], [6]]
    batches = encode_batches(tokenizer, token_ids, short_width=48)

    positions, encoded = batches[0]
    assert positions == [2, 0, 1]
    assert encoded["input_ids"].tolist() == [
        [0, 0, 6],
        [0, 1, 2],
        [3, 4, 5],
    ]
    assert encoded["attention_mask"].tolist() == [
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]


def test_workload_report_tracks_padding_and_peak_shape() -> None:
    batches = [
        (
            [0, 1],
            {
                "input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            },
        ),
        (
            [2],
            {
                "input_ids": torch.tensor([[6, 7, 8, 9]]),
            },
        ),
    ]
    report = workload_report(batches, [[1, 2], [3, 4, 5], [6, 7, 8, 9]])

    assert report["input_tokens"] == 9
    assert report["padded_tokens"] == 10
    assert report["padding_tokens"] == 1
    assert report["traces"] == 2
    assert report["max_padded_tokens_per_trace"] == 6


def test_dynamic_batches_obey_row_and_padded_token_caps() -> None:
    lengths = [200] * 56 + [500] * 56 + [800] * 40 + [1_800] * 16
    batches = dynamic_position_batches(
        lengths,
        row_cap=56,
        padded_token_budget=28_800,
    )

    assert sorted(position for batch in batches for position in batch) == list(
        range(len(lengths))
    )
    assert all(len(batch) <= 56 for batch in batches)
    assert all(
        len(batch) * max(lengths[position] for position in batch) <= 28_800
        for batch in batches
    )
    assert any(len(batch) not in {16, 32, 48, 56} for batch in batches)
