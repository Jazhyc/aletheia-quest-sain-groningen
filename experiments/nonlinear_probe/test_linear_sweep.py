"""
Tests for the pure helpers in linear_sweep.py (no GPU, no network, no npz files).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linear_sweep import (
    CV_EVAL_LABEL,
    PER_DATASET_COLUMNS,
    SWEEP_COLUMNS,
    base_model_family,
    compute_metrics,
    discover_cache_files,
    enumerate_configs,
    has_both_classes,
    is_limit_cache,
    is_tokens_cache,
    meta_model_id,
    parse_layers_arg,
    parse_scenario,
    per_dataset_metrics,
    rows_to_frame,
    top_configs,
    valid_label_mask,
)


class TestIsLimitCache:

    def test_flags_limit_suffix(self) -> None:
        assert is_limit_cache("aletheias-quest__dev-varied-deception-Qwen3.5-27B-None.limit32.npz")

    def test_full_cache_is_not_limit(self) -> None:
        assert not is_limit_cache("aletheias-quest__dev-varied-deception-Qwen3.5-27B-None.npz")


class TestIsTokensCache:

    def test_flags_tokens_suffix(self) -> None:
        assert is_tokens_cache("aletheias-quest__dev-varied-deception-Qwen3.5-27B-None.tokens.npz")

    def test_pooled_cache_is_not_tokens(self) -> None:
        assert not is_tokens_cache("aletheias-quest__dev-varied-deception-Qwen3.5-27B-None.npz")


class TestDiscoverCacheFiles:

    def test_skips_limit_files_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "full.npz").touch()
        (tmp_path / "smoke.limit32.npz").touch()
        found = discover_cache_files(tmp_path)
        assert [path.name for path in found] == ["full.npz"]

    def test_include_limit_returns_everything(self, tmp_path: Path) -> None:
        (tmp_path / "full.npz").touch()
        (tmp_path / "smoke.limit32.npz").touch()
        found = discover_cache_files(tmp_path, include_limit=True)
        assert len(found) == 2

    def test_ignores_non_npz_files(self, tmp_path: Path) -> None:
        (tmp_path / "full.npz").touch()
        (tmp_path / "notes.txt").touch()
        found = discover_cache_files(tmp_path)
        assert [path.name for path in found] == ["full.npz"]

    def test_skips_tokens_files_even_with_include_limit(self, tmp_path: Path) -> None:
        (tmp_path / "full.npz").touch()
        (tmp_path / "full.tokens.npz").touch()
        (tmp_path / "smoke.tokens.limit32.npz").touch()
        found = discover_cache_files(tmp_path, include_limit=True)
        assert [path.name for path in found] == ["full.npz"]


class TestParseScenario:

    def test_varied(self) -> None:
        assert parse_scenario("aletheias-quest/dev-varied-deception-Qwen3.5-27B-None") == "varied"

    def test_instructed(self) -> None:
        dataset = "aletheias-quest/dev-instructed-deception-gemma-3-27b-it-None"
        assert parse_scenario(dataset) == "instructed"

    def test_organism_suffix_does_not_confuse_parsing(self) -> None:
        dataset = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1"
        assert parse_scenario(dataset) == "varied"

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="could not parse scenario"):
            parse_scenario("aletheias-quest/dev-weird-deception-Qwen3.5-27B-None")


class TestBaseModelFamily:

    def test_qwen(self) -> None:
        assert base_model_family("Qwen/Qwen3.5-27B") == "qwen"

    def test_gemma(self) -> None:
        assert base_model_family("google/gemma-3-27b-it") == "gemma"

    def test_unknown_family_falls_back_to_last_path_segment(self) -> None:
        assert base_model_family("mistralai/Mistral-7B") == "mistral-7b"


class TestMetaModelId:

    def test_prefers_model_id_key(self) -> None:
        assert meta_model_id({"model_id": "a", "model": "b"}) == "a"

    def test_falls_back_to_model_key(self) -> None:
        assert meta_model_id({"model": "google/gemma-3-27b-it"}) == "google/gemma-3-27b-it"

    def test_missing_both_raises(self) -> None:
        with pytest.raises(KeyError):
            meta_model_id({})


class TestValidLabelMask:

    def test_bool_labels_are_all_valid(self) -> None:
        labels = np.array([True, False, True])
        assert valid_label_mask(labels).tolist() == [True, True, True]

    def test_negative_int_labels_are_invalid(self) -> None:
        labels = np.array([1, 0, -1, -1, 1])
        assert valid_label_mask(labels).tolist() == [True, True, False, False, True]


class TestEnumerateConfigs:

    def test_cross_product_and_dedup(self) -> None:
        configs = enumerate_configs(["qwen", "qwen", "gemma"], [4, 2], ["mean", "last"])
        assert configs == [
            ("gemma", 2, "mean"), ("gemma", 2, "last"),
            ("gemma", 4, "mean"), ("gemma", 4, "last"),
            ("qwen", 2, "mean"), ("qwen", 2, "last"),
            ("qwen", 4, "mean"), ("qwen", 4, "last"),
        ]


class TestHasBothClasses:

    def test_two_classes(self) -> None:
        assert has_both_classes(np.array([0, 1, 0, 1]))

    def test_single_class(self) -> None:
        assert not has_both_classes(np.array([0, 0, 0]))

    def test_empty(self) -> None:
        assert not has_both_classes(np.array([]))


class TestComputeMetrics:

    def test_perfect_separation(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_metrics(labels, scores)
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["balanced_accuracy"] == pytest.approx(1.0)

    def test_single_class_labels_give_nan_auroc(self) -> None:
        labels = np.array([0, 0, 0])
        scores = np.array([0.1, 0.6, 0.9])
        metrics = compute_metrics(labels, scores)
        assert np.isnan(metrics["auroc"])
        assert not np.isnan(metrics["balanced_accuracy"])


class TestPerDatasetMetrics:

    def test_splits_by_dataset_id(self) -> None:
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.9, 0.1])
        dataset_ids = np.array(["a", "a", "b", "b"], dtype=object)
        rows = per_dataset_metrics(labels, scores, dataset_ids, base_model="qwen", layer=4,
                                   pooling="mean", train_scenario="varied", eval_scenario="instructed")
        assert {row["dataset"] for row in rows} == {"a", "b"}
        by_dataset = {row["dataset"]: row for row in rows}
        assert by_dataset["a"]["balanced_accuracy"] == pytest.approx(1.0)
        assert by_dataset["b"]["balanced_accuracy"] == pytest.approx(0.0)
        assert by_dataset["a"]["n_eval"] == 2
        assert all(row["base_model"] == "qwen" for row in rows)


class TestRowsToFrame:

    def test_empty_rows_keep_columns(self) -> None:
        frame = rows_to_frame([], SWEEP_COLUMNS)
        assert list(frame.columns) == SWEEP_COLUMNS
        assert len(frame) == 0

    def test_nonempty_rows_reordered_to_columns(self) -> None:
        rows = [{
            "eval_scenario": "instructed", "base_model": "qwen", "layer": 4, "pooling": "mean",
            "train_scenario": "varied", "auroc": 0.9, "balanced_accuracy": 0.8,
            "n_train": 10, "n_eval": 5,
        }]
        frame = rows_to_frame(rows, SWEEP_COLUMNS)
        assert list(frame.columns) == SWEEP_COLUMNS
        assert frame.iloc[0]["base_model"] == "qwen"


class TestTopConfigs:

    def test_ranks_by_mean_balanced_accuracy(self) -> None:
        rows = [
            dict(base_model="qwen", layer=4, pooling="mean", train_scenario="varied",
                eval_scenario="instructed", auroc=0.9, balanced_accuracy=0.9, n_train=10, n_eval=10),
            dict(base_model="qwen", layer=4, pooling="mean", train_scenario="instructed",
                eval_scenario="varied", auroc=0.7, balanced_accuracy=0.7, n_train=10, n_eval=10),
            dict(base_model="qwen", layer=8, pooling="mean", train_scenario="varied",
                eval_scenario=CV_EVAL_LABEL, auroc=0.6, balanced_accuracy=0.6, n_train=10, n_eval=10),
        ]
        frame = rows_to_frame(rows, SWEEP_COLUMNS)
        summary = top_configs(frame, top_n=10)
        assert summary.iloc[0]["layer"] == 4
        assert summary.iloc[0]["mean_balanced_accuracy"] == pytest.approx(0.8)
        assert summary.iloc[1]["layer"] == 8

    def test_empty_frame_returns_empty_summary(self) -> None:
        frame = rows_to_frame([], SWEEP_COLUMNS)
        summary = top_configs(frame, top_n=10)
        assert summary.empty

    def test_top_n_limits_rows(self) -> None:
        rows = [
            dict(base_model="qwen", layer=layer, pooling="mean", train_scenario="varied",
                eval_scenario=CV_EVAL_LABEL, auroc=0.5, balanced_accuracy=layer / 10,
                n_train=10, n_eval=10)
            for layer in range(20)
        ]
        frame = rows_to_frame(rows, SWEEP_COLUMNS)
        summary = top_configs(frame, top_n=3)
        assert len(summary) == 3
        assert summary.iloc[0]["layer"] == 19


class TestParseLayersArg:

    def test_none_spec(self) -> None:
        assert parse_layers_arg(None) is None

    def test_empty_spec(self) -> None:
        assert parse_layers_arg("") is None

    def test_comma_list(self) -> None:
        assert parse_layers_arg("0,4,8") == [0, 4, 8]


def test_per_dataset_columns_are_distinct_from_sweep_columns() -> None:
    assert "dataset" in PER_DATASET_COLUMNS
    assert "dataset" not in SWEEP_COLUMNS
