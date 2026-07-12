"""
Tests for the pure helpers in nonlinear_sweep.py (no GPU, no network, no npz files).
"""

from pathlib import Path

import numpy as np
import pytest

from linear_sweep import CacheFile, rows_to_frame
from nonlinear_sweep import (
    HOLDOUT_COLUMNS,
    common_layers,
    fit_probe,
    holdout_summary,
    parse_hidden_arg,
    parse_organism,
    probe_label,
    scenarios_of,
)


def make_cache(dataset: str, scenario: str, layers: list[int],
               model_id: str = "Qwen/Qwen3.5-27B") -> CacheFile:
    labels = np.array([0, 1])
    return CacheFile(
        path=Path("/nonexistent.npz"), dataset=dataset, scenario=scenario,
        base_model="qwen", model_id=model_id, layers=layers,
        labels=labels, label_mask=np.ones(2, dtype=bool),
    )


class TestParseOrganism:

    def test_none_organism(self) -> None:
        dataset = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"
        assert parse_organism(dataset, "Qwen/Qwen3.5-27B") == "None"

    def test_strips_base_model_echo_with_trailing_index(self) -> None:
        dataset = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3"
        assert parse_organism(dataset, "Qwen/Qwen3.5-27B") == "a-mo-3"

    def test_strips_base_model_echo_without_index(self) -> None:
        dataset = "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b"
        assert parse_organism(dataset, "Qwen/Qwen3.5-27B") == "b-mo"

    def test_gemma_organism(self) -> None:
        dataset = "aletheias-quest/dev-instructed-deception-gemma-3-27b-it-g-st-gemma-3-27b-it-2"
        assert parse_organism(dataset, "google/gemma-3-27b-it") == "g-st-2"

    def test_wrong_model_token_raises(self) -> None:
        dataset = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"
        with pytest.raises(ValueError, match="could not parse organism"):
            parse_organism(dataset, "google/gemma-3-27b-it")

    def test_same_organism_across_scenarios(self) -> None:
        varied = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b"
        instructed = "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b"
        assert parse_organism(varied, "Qwen/Qwen3.5-27B") \
            == parse_organism(instructed, "Qwen/Qwen3.5-27B") == "c-mo"


class TestProbeLabel:

    def test_logistic_ignores_hidden(self) -> None:
        assert probe_label("logistic", (512,)) == "logistic"

    def test_mlp_single_hidden(self) -> None:
        assert probe_label("mlp", (512,)) == "mlp-512"

    def test_mlp_two_hidden(self) -> None:
        assert probe_label("mlp", (512, 128)) == "mlp-512x128"


class TestParseHiddenArg:

    def test_single_size(self) -> None:
        assert parse_hidden_arg("512") == (512,)

    def test_two_sizes(self) -> None:
        assert parse_hidden_arg("512,128") == (512, 128)


class TestFitProbe:

    def make_separable_data(self, n_rows: int = 400) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(n_rows, 6)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(np.int64)
        features[:, 0] += labels * 2.0
        return features, labels

    def test_logistic_fits_and_scores(self) -> None:
        features, labels = self.make_separable_data()
        pipeline = fit_probe(features, labels, "logistic", (16,), 1e-3, 1.0, 0)
        scores = pipeline.predict_proba(features)[:, 1]
        assert scores.shape == (400,)
        assert ((scores >= 0.5).astype(int) == labels).mean() > 0.9

    def test_mlp_fits_and_scores(self) -> None:
        features, labels = self.make_separable_data()
        pipeline = fit_probe(features, labels, "mlp", (16,), 1e-3, 1.0, 0)
        scores = pipeline.predict_proba(features)[:, 1]
        assert scores.shape == (400,)
        assert ((scores >= 0.5).astype(int) == labels).mean() > 0.9

    def test_unknown_probe_raises(self) -> None:
        features, labels = self.make_separable_data()
        with pytest.raises(ValueError, match="unknown probe"):
            fit_probe(features, labels, "svm", (16,), 1e-3, 1.0, 0)


class TestScenariosOf:

    def test_sorted_distinct(self) -> None:
        caches = [
            make_cache("d1", "varied", [0]),
            make_cache("d2", "instructed", [0]),
            make_cache("d3", "varied", [0]),
        ]
        assert scenarios_of(caches) == ["instructed", "varied"]


class TestCommonLayers:

    def test_intersects_across_caches(self) -> None:
        caches = [make_cache("d1", "varied", [0, 2, 4]), make_cache("d2", "varied", [2, 4, 6])]
        assert common_layers(caches, None) == [2, 4]

    def test_override_restricts_further(self) -> None:
        caches = [make_cache("d1", "varied", [0, 2, 4]), make_cache("d2", "varied", [2, 4, 6])]
        assert common_layers(caches, [4, 8]) == [4]


class TestHoldoutSummary:

    def test_pivots_probes_to_columns(self) -> None:
        rows = [
            dict(base_model="qwen", probe="logistic", layer=46, pooling="concat",
                 holdout_organism="a-mo-3", auroc=0.8, balanced_accuracy=0.7,
                 n_train=100, n_eval=20),
            dict(base_model="qwen", probe="mlp-512", layer=46, pooling="concat",
                 holdout_organism="a-mo-3", auroc=0.9, balanced_accuracy=0.85,
                 n_train=100, n_eval=20),
        ]
        summary = holdout_summary(rows_to_frame(rows, HOLDOUT_COLUMNS))
        assert list(summary.columns) == ["logistic", "mlp-512"]
        assert summary.iloc[0]["mlp-512"] == pytest.approx(0.85)

    def test_empty_frame_gives_empty_summary(self) -> None:
        summary = holdout_summary(rows_to_frame([], HOLDOUT_COLUMNS))
        assert summary.empty
