"""
Tests for the pure helpers in extract_activations.py (no NDIF, no torch).
"""

import pytest

from extract_activations import budget_batches, parse_layers


class TestParseLayers:

    def test_slice_every_second(self) -> None:
        assert parse_layers("::2", 6) == [0, 2, 4]

    def test_slice_with_bounds(self) -> None:
        assert parse_layers("8:20:4", 64) == [8, 12, 16]

    def test_comma_list(self) -> None:
        assert parse_layers("5,17,29", 32) == [5, 17, 29]

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            parse_layers("5,40", 32)


class TestBudgetBatches:

    def test_covers_every_position_exactly_once(self) -> None:
        lengths = [300, 10, 250, 40, 512, 90]
        batches = budget_batches(lengths, token_budget=1024, max_batch=4)
        flat = sorted(position for batch in batches for position in batch)
        assert flat == list(range(len(lengths)))

    def test_product_stays_under_budget(self) -> None:
        lengths = [512, 256, 128, 64, 500, 490, 32, 400]
        token_budget = 1024
        for batch in budget_batches(lengths, token_budget, max_batch=32):
            width = max(lengths[position] for position in batch)
            assert len(batch) * width <= token_budget

    def test_respects_max_batch(self) -> None:
        lengths = [8] * 100
        batches = budget_batches(lengths, token_budget=10_000, max_batch=16)
        assert all(len(batch) <= 16 for batch in batches)

    def test_batches_are_length_sorted(self) -> None:
        lengths = [50, 10, 30, 20, 40]
        batches = budget_batches(lengths, token_budget=60, max_batch=2)
        widths = [max(lengths[position] for position in batch) for batch in batches]
        assert widths == sorted(widths)
