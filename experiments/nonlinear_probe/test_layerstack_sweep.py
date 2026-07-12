"""
Tests for torch_probes.py and the layer-stack feature loading in
layerstack_sweep.py (small synthetic data only, no cached activations).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from layerstack_sweep import concat_stacked_features, stacked_features
from linear_sweep import load_cache_file
from torch_probes import (
    CNNLayerProbe,
    TorchProbe,
    TransformerLayerProbe,
    build_probe_model,
    streaming_moments,
)


def write_synthetic_cache(path: Path, n_rows: int = 8, layers: tuple[int, ...] = (0, 2, 4),
                          hidden_dim: int = 6) -> None:
    rng = np.random.default_rng(0)
    arrays = {
        f"{pooling}_L{layer}": rng.normal(size=(n_rows, hidden_dim)).astype(np.float16)
        for pooling in ("mean", "last") for layer in layers
    }
    meta = {"dataset": "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None",
            "model": "Qwen/Qwen3.5-27B", "layers": list(layers)}
    np.savez(path, **arrays, deceptive=np.arange(n_rows) % 2 == 0,
             index=np.arange(n_rows), meta=json.dumps(meta))


class TestProbeModels:

    def test_cnn_forward_shape(self) -> None:
        probe = CNNLayerProbe(hidden_dim=6, channels=4)
        logits = probe(torch.randn(5, 3, 6))
        assert logits.shape == (5,)

    def test_transformer_forward_shape(self) -> None:
        probe = TransformerLayerProbe(hidden_dim=6, n_layers=3, d_model=8, n_heads=2)
        logits = probe(torch.randn(5, 3, 6))
        assert logits.shape == (5,)

    def test_build_probe_model_dispatch(self) -> None:
        assert isinstance(build_probe_model("cnn", 6, 3), CNNLayerProbe)
        assert isinstance(build_probe_model("transformer", 6, 3), TransformerLayerProbe)

    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown architecture"):
            build_probe_model("rnn", 6, 3)


class TestStreamingMoments:

    def test_matches_direct_computation(self) -> None:
        rng = np.random.default_rng(1)
        features = rng.normal(size=(50, 3, 4)).astype(np.float16)
        mean, std = streaming_moments(features, chunk_rows=7)
        direct = features.astype(np.float64)
        np.testing.assert_allclose(mean, direct.mean(axis=0), atol=1e-3)
        np.testing.assert_allclose(std, direct.std(axis=0), atol=1e-3)


class TestTorchProbe:

    def make_separable_stacks(self, n_rows: int = 400) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(n_rows, 3, 6)).astype(np.float16)
        labels = (features[:, 1, :].mean(axis=1) > 0).astype(np.int64)
        features[:, 1, :] += (labels * 1.5)[:, None].astype(np.float16)
        return features, labels

    @pytest.mark.parametrize("architecture", ["cnn", "transformer"])
    def test_fit_and_score_separable_data(self, architecture: str) -> None:
        features, labels = self.make_separable_stacks()
        probe = TorchProbe(architecture, seed=0, device="cpu", max_epochs=30)
        scores = probe.fit(features, labels).predict_proba(features)[:, 1]
        assert scores.shape == (400,)
        assert ((scores >= 0.5).astype(int) == labels).mean() > 0.9

    def test_predict_proba_layout(self) -> None:
        features, labels = self.make_separable_stacks(100)
        probe = TorchProbe("cnn", seed=0, device="cpu", max_epochs=5)
        probabilities = probe.fit(features, labels).predict_proba(features)
        assert probabilities.shape == (100, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


class TestStackedFeatures:

    def test_shape_and_order(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "synthetic.npz"
        write_synthetic_cache(cache_path)
        cache = load_cache_file(cache_path)
        stack = stacked_features(cache, "mean", layer_step=1)
        assert stack.shape == (8, 3, 6)
        assert stack.dtype == np.float16

    def test_layer_step_subsamples(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "synthetic.npz"
        write_synthetic_cache(cache_path)
        cache = load_cache_file(cache_path)
        stack = stacked_features(cache, "mean", layer_step=2)
        assert stack.shape == (8, 2, 6)

    def test_concat_keeps_labels_aligned(self, tmp_path: Path) -> None:
        first_path = tmp_path / "first.npz"
        second_path = tmp_path / "second.npz"
        write_synthetic_cache(first_path)
        write_synthetic_cache(second_path)
        caches = [load_cache_file(first_path), load_cache_file(second_path)]
        features, labels, dataset_ids = concat_stacked_features(caches, "mean", 1)
        assert features.shape == (16, 3, 6)
        assert len(labels) == len(dataset_ids) == 16
