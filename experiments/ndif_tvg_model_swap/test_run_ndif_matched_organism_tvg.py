from __future__ import annotations

import pytest

from experiments.ndif_tvg_model_swap.run_ndif_matched_organism_tvg import (
    batch_size_for,
    breakdown,
    family_for,
    safe_group_name,
    validate_cached_group,
    validate_group,
)


def test_family_batch_sizes_and_safe_names() -> None:
    assert batch_size_for("Qwen/Qwen3.5-27B") == 16
    assert batch_size_for("google/gemma-3-27b-it") == 8
    assert batch_size_for("nvidia/NVIDIA-Nemotron-3-Super") == 1
    assert family_for("Qwen/Qwen3.5-27B") == "qwen"
    assert safe_group_name(
        "Qwen/Qwen3.5-27B",
        "aletheias-quest/a-mo",
    ) == "Qwen_Qwen3.5-27B__aletheias-quest_a-mo"


def test_group_must_have_one_exact_organism() -> None:
    rows = [
        {"model": "Qwen/Qwen3.5-27B", "lora": None},
        {"model": "Qwen/Qwen3.5-27B", "lora": ""},
    ]
    assert validate_group(rows) == ("Qwen/Qwen3.5-27B", None)
    rows[1]["lora"] = "adapter"
    with pytest.raises(ValueError, match="multiple organisms"):
        validate_group(rows)


def test_cache_validation_includes_lora() -> None:
    cached = {
        "model": "model",
        "lora": "adapter",
        "keys": [["dataset", "1"]],
        "prompt_sha256": ["hash"],
        "scores": [0.25],
    }
    validate_cached_group(
        cached,
        model_id="model",
        lora_id="adapter",
        keys=[["dataset", "1"]],
        hashes=["hash"],
    )
    with pytest.raises(ValueError, match="LoRA mismatch"):
        validate_cached_group(
            cached,
            model_id="model",
            lora_id=None,
            keys=[["dataset", "1"]],
            hashes=["hash"],
        )


def test_breakdown_remains_macro_by_dataset() -> None:
    rows = [
        {"dataset": "a", "label": 0, "score": 0.1, "family": "qwen"},
        {"dataset": "a", "label": 1, "score": 0.9, "family": "qwen"},
        {"dataset": "b", "label": 0, "score": 0.8, "family": "qwen"},
        {"dataset": "b", "label": 1, "score": 0.7, "family": "qwen"},
    ]
    result = breakdown(rows, lambda row: row["family"])
    assert result["qwen"]["dataset_units"] == 2
    assert result["qwen"]["metrics"]["auroc"] == pytest.approx(0.5)
