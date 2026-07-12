"""
Tests for token_probes.py and the token-cache loading in token_sweep.py
(small synthetic data only, no cached activations, device='cpu' throughout).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from token_probes import (
    AttentionTokenProbe,
    CNNTokenProbe,
    TokenProbe,
    TransformerTokenProbe,
    build_token_probe_model,
    pack_length_sorted_batches,
    streaming_token_moments,
)
from token_sweep import (
    TOKENS_POOLING_LABEL,
    concat_token_features,
    load_token_cache_file,
    load_token_features,
)


def write_synthetic_token_cache(
        path: Path, lengths: list[int], layers: tuple[int, ...] = (0, 2, 4),
        hidden_dim: int = 6, dataset: str = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None",
        model_id: str = "Qwen/Qwen3.5-27B", seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    total_tokens = int(offsets[-1])
    arrays = {
        f"tokens_L{layer}": rng.normal(size=(total_tokens, hidden_dim)).astype(np.float16)
        for layer in layers
    }
    meta = {"dataset": dataset, "model": model_id, "layers": list(layers), "mode": "tokens"}
    np.savez(
        path, **arrays, token_offsets=offsets, index=np.arange(len(lengths)),
        deceptive=np.arange(len(lengths)) % 2 == 0,
        response_tokens=np.asarray(lengths, dtype=np.int32), meta=json.dumps(meta),
    )


class TestProbeModelForwardShapes:

    def test_attention_forward_shape(self) -> None:
        probe = AttentionTokenProbe(hidden_dim=6, d_model=8)
        padded_tokens = torch.randn(5, 4, 6)
        padding_mask = torch.ones(5, 4, dtype=torch.bool)
        logits = probe(padded_tokens, padding_mask)
        assert logits.shape == (5,)

    def test_cnn_forward_shape(self) -> None:
        probe = CNNTokenProbe(hidden_dim=6, channels=4, kernel_size=3)
        padded_tokens = torch.randn(5, 4, 6)
        padding_mask = torch.ones(5, 4, dtype=torch.bool)
        logits = probe(padded_tokens, padding_mask)
        assert logits.shape == (5,)

    def test_transformer_forward_shape(self) -> None:
        probe = TransformerTokenProbe(hidden_dim=6, d_model=8, n_heads=2, dim_feedforward=16)
        padded_tokens = torch.randn(5, 4, 6)
        padding_mask = torch.ones(5, 4, dtype=torch.bool)
        logits = probe(padded_tokens, padding_mask)
        assert logits.shape == (5,)

    def test_build_token_probe_model_dispatch(self) -> None:
        assert isinstance(build_token_probe_model("attention", 6), AttentionTokenProbe)
        assert isinstance(build_token_probe_model("cnn", 6), CNNTokenProbe)
        assert isinstance(build_token_probe_model("transformer", 6), TransformerTokenProbe)

    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown architecture"):
            build_token_probe_model("rnn", 6)

    @pytest.mark.parametrize("architecture", ["attention", "cnn", "transformer"])
    def test_padding_does_not_affect_output(self, architecture: str) -> None:
        torch.manual_seed(0)
        probe = build_token_probe_model(architecture, hidden_dim=6)
        probe.eval()
        real_length = 3
        real_tokens = torch.randn(1, real_length, 6)
        unpadded_mask = torch.ones(1, real_length, dtype=torch.bool)

        padded_width = real_length + 2
        padded_tokens = torch.zeros(1, padded_width, 6)
        padded_tokens[:, :real_length] = real_tokens
        padded_mask = torch.zeros(1, padded_width, dtype=torch.bool)
        padded_mask[:, :real_length] = True

        with torch.no_grad():
            unpadded_logits = probe(real_tokens, unpadded_mask)
            padded_logits = probe(padded_tokens, padded_mask)
        torch.testing.assert_close(unpadded_logits, padded_logits, atol=1e-4, rtol=1e-3)


class TestPackLengthSortedBatches:

    def test_covers_every_row_exactly_once(self) -> None:
        lengths = [5, 12, 3, 8, 20, 1, 7, 15]
        batches = pack_length_sorted_batches(lengths, token_budget=32)
        flattened = sorted(position for batch in batches for position in batch)
        assert flattened == list(range(len(lengths)))

    def test_respects_token_budget(self) -> None:
        lengths = [5, 12, 3, 8, 20, 1, 7, 15]
        token_budget = 32
        batches = pack_length_sorted_batches(lengths, token_budget)
        for batch in batches:
            max_len = max(lengths[position] for position in batch)
            if len(batch) > 1:
                assert len(batch) * max_len <= token_budget
            else:
                assert max_len <= token_budget or len(batch) == 1

    def test_single_example_over_budget_is_its_own_batch(self) -> None:
        lengths = [50]
        batches = pack_length_sorted_batches(lengths, token_budget=10)
        assert batches == [[0]]

    def test_empty_lengths(self) -> None:
        assert pack_length_sorted_batches([], token_budget=32) == []


class TestStreamingTokenMoments:

    def test_matches_direct_computation_over_subset(self) -> None:
        rng = np.random.default_rng(1)
        lengths = [3, 5, 2, 4]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        flat_features = torch.from_numpy(
            rng.normal(size=(int(offsets[-1]), 4)).astype(np.float16))
        row_indices = np.array([0, 2, 3])
        mean, std = streaming_token_moments(flat_features, offsets, row_indices, chunk_rows=2)

        selected_tokens = torch.cat(
            [flat_features[offsets[row]:offsets[row + 1]] for row in row_indices], dim=0,
        ).to(torch.float64)
        expected_mean = selected_tokens.mean(dim=0)
        expected_std = selected_tokens.std(dim=0, unbiased=False)
        np.testing.assert_allclose(mean.numpy(), expected_mean.numpy(), atol=1e-2)
        np.testing.assert_allclose(std.numpy(), expected_std.numpy(), atol=1e-2)


class TestTokenProbe:

    def make_separable_tokens(
            self, n_rows: int = 300, hidden_dim: int = 6,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        lengths = rng.integers(low=2, high=6, size=n_rows)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        labels = (rng.random(n_rows) > 0.5).astype(np.int64)
        flat_features = rng.normal(size=(int(offsets[-1]), hidden_dim)).astype(np.float32)
        for row in range(n_rows):
            if labels[row] == 1:
                flat_features[offsets[row]:offsets[row + 1]] += 3.0
        return torch.from_numpy(flat_features.astype(np.float16)), offsets, labels

    @pytest.mark.parametrize("architecture", ["attention", "cnn"])
    def test_fit_and_score_separable_data(self, architecture: str) -> None:
        flat_features, offsets, labels = self.make_separable_tokens()
        probe = TokenProbe(architecture, seed=0, device="cpu", max_epochs=30)
        probe.fit(flat_features, offsets, labels)
        scores = probe.predict_proba(flat_features, offsets)[:, 1]
        assert scores.shape == (300,)
        assert ((scores >= 0.5).astype(int) == labels).mean() > 0.9

    def test_transformer_fit_and_score_separable_data(self) -> None:
        flat_features, offsets, labels = self.make_separable_tokens()
        probe = TokenProbe("transformer", seed=0, device="cpu", max_epochs=40)
        probe.fit(flat_features, offsets, labels)
        scores = probe.predict_proba(flat_features, offsets)[:, 1]
        assert ((scores >= 0.5).astype(int) == labels).mean() > 0.8

    def test_predict_proba_layout(self) -> None:
        flat_features, offsets, labels = self.make_separable_tokens(80)
        probe = TokenProbe("cnn", seed=0, device="cpu", max_epochs=5)
        probabilities = probe.fit(flat_features, offsets, labels).predict_proba(
            flat_features, offsets)
        assert probabilities.shape == (80, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_row_order_independent_of_batch_packing(self) -> None:
        flat_features, offsets, labels = self.make_separable_tokens(60)
        probe = TokenProbe("cnn", seed=0, device="cpu", max_epochs=5, batch_token_budget=64)
        probe.fit(flat_features, offsets, labels)
        small_budget_scores = probe.predict_proba(flat_features, offsets)[:, 1]
        probe.batch_token_budget = 4096
        large_budget_scores = probe.predict_proba(flat_features, offsets)[:, 1]
        np.testing.assert_allclose(small_budget_scores, large_budget_scores, atol=1e-5)


class TestTokenCacheLoading:

    def test_load_token_cache_file_metadata(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "synthetic.tokens.npz"
        write_synthetic_token_cache(cache_path, lengths=[3, 5, 2, 4])
        cache = load_token_cache_file(cache_path)
        assert cache is not None
        assert cache.dataset == "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"
        assert cache.scenario == "varied"
        assert cache.base_model == "qwen"
        assert cache.layers == [0, 2, 4]
        assert list(cache.labels) == [1, 0, 1, 0]

    def test_load_token_cache_file_without_labels_returns_none(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "unlabeled.tokens.npz"
        lengths = [3, 5]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        meta = {"dataset": "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None",
                "model": "Qwen/Qwen3.5-27B", "layers": [0], "mode": "tokens"}
        np.savez(
            cache_path, tokens_L0=np.zeros((int(offsets[-1]), 4), dtype=np.float16),
            token_offsets=offsets, index=np.arange(len(lengths)), meta=json.dumps(meta),
        )
        assert load_token_cache_file(cache_path) is None

    def test_load_token_features_shape(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "synthetic.tokens.npz"
        write_synthetic_token_cache(cache_path, lengths=[3, 5, 2, 4], hidden_dim=6)
        cache = load_token_cache_file(cache_path)
        flat_features, offsets = load_token_features(cache, layer=2, device="cpu")
        assert flat_features.shape == (14, 6)
        assert flat_features.dtype == torch.float16
        np.testing.assert_array_equal(offsets, [0, 3, 8, 10, 14])

    def test_concat_shifts_offsets_and_keeps_labels_aligned(self, tmp_path: Path) -> None:
        first_path = tmp_path / "first.tokens.npz"
        second_path = tmp_path / "second.tokens.npz"
        write_synthetic_token_cache(
            first_path, lengths=[3, 5], dataset="aletheias-quest/dev-varied-deception-Qwen3.5-27B-None",
            seed=0)
        write_synthetic_token_cache(
            second_path, lengths=[2, 4, 1],
            dataset="aletheias-quest/dev-instructed-deception-Qwen3.5-27B-None", seed=1)
        caches = [load_token_cache_file(first_path), load_token_cache_file(second_path)]
        flat_features, offsets, labels, dataset_ids = concat_token_features(
            caches, layer=0, device="cpu")

        assert flat_features.shape == (3 + 5 + 2 + 4 + 1, 6)
        np.testing.assert_array_equal(offsets, [0, 3, 8, 10, 14, 15])
        assert list(labels) == [1, 0, 1, 0, 1]
        assert list(dataset_ids[:2]) == [caches[0].dataset] * 2
        assert list(dataset_ids[2:]) == [caches[1].dataset] * 3

    def test_pooling_label_constant(self) -> None:
        assert TOKENS_POOLING_LABEL == "tokens"
