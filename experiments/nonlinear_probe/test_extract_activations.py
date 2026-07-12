"""
Tests for the pure helpers in extract_activations.py (no NDIF).
"""

import numpy as np
import pytest

from extract_activations import assemble_token_features, budget_batches, parse_layers


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


class TestAssembleTokenFeatures:

    def test_out_of_order_batches_reordered_to_dataset_order(self) -> None:
        # dataset order: position 0 has 3 tokens, position 1 has 2, position 2
        # has 4; batches traverse them out of order as [2, 0] then [1].
        span_lengths = [3, 2, 4]
        batches = [[2, 0], [1]]
        flat_features = np.array([[
            [20], [21], [22], [23],   # position 2's 4 tokens
            [0], [1], [2],            # position 0's 3 tokens
            [10], [11],               # position 1's 2 tokens
        ]], dtype=np.float32)

        layer_features, token_offsets = assemble_token_features(flat_features, batches, span_lengths)

        expected = np.array([[0], [1], [2], [10], [11], [20], [21], [22], [23]], dtype=np.float32)
        assert np.array_equal(layer_features[0], expected)
        assert token_offsets.tolist() == [0, 3, 5, 9]

    def test_offsets_sum_matches_total_tokens(self) -> None:
        span_lengths = [3, 2, 4]
        batches = [[2, 0], [1]]
        flat_features = np.zeros((1, sum(span_lengths), 1), dtype=np.float32)

        _, token_offsets = assemble_token_features(flat_features, batches, span_lengths)

        assert token_offsets[0] == 0
        assert token_offsets[-1] == sum(span_lengths)

    def test_multi_layer_arrays_stay_aligned(self) -> None:
        span_lengths = [3, 2, 4]
        batches = [[2, 0], [1]]
        layer0 = [20, 21, 22, 23, 0, 1, 2, 10, 11]
        layer1 = [value + 100 for value in layer0]
        flat_features = np.array([
            [[value] for value in layer0],
            [[value] for value in layer1],
        ], dtype=np.float32)

        layer_features, token_offsets = assemble_token_features(flat_features, batches, span_lengths)

        expected_layer0 = np.array([[0], [1], [2], [10], [11], [20], [21], [22], [23]], dtype=np.float32)
        assert np.array_equal(layer_features[0], expected_layer0)
        assert np.array_equal(layer_features[1], expected_layer0 + 100)
        assert token_offsets.tolist() == [0, 3, 5, 9]
