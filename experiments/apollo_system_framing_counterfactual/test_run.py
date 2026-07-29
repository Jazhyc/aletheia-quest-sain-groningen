from __future__ import annotations

import pandas as pd
import pytest

from experiments.apollo_system_framing_counterfactual.run import (
    CONDITIONS,
    cache_keys,
    paired_comparison,
    validate_cached_group,
)
from experiments.phoenix_system_framing_counterfactual.run import (
    framing_messages,
)


def test_framing_preserves_answer_and_native_reasoning() -> None:
    messages = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "native trace",
        },
    ]
    changed = framing_messages(messages, "competition_decommission")
    assert changed[0]["content"] != "old"
    assert changed[1:] == messages[1:]
    assert messages[0]["content"] == "old"


def test_cache_keys_distinguish_conditions_and_splits() -> None:
    rows = [
        {
            "split": split,
            "dataset": "dataset",
            "index": 7,
            "condition": condition,
        }
        for split in ("validation", "test")
        for condition in CONDITIONS
    ]
    keys = cache_keys(rows)
    assert len(keys) == len(set(map(tuple, keys))) == 10


def test_cache_validation_binds_rendered_prompts() -> None:
    cached = {
        "model": "model",
        "lora": "adapter",
        "keys": [["test", "dataset", "1", "original"]],
        "prompt_sha256": ["hash"],
        "scores": [0.2],
    }
    validate_cached_group(
        cached,
        model_id="model",
        lora_id="adapter",
        keys=cached["keys"],
        hashes=["hash"],
    )
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_cached_group(
            cached,
            model_id="model",
            lora_id="adapter",
            keys=cached["keys"],
            hashes=["different"],
        )


def test_paired_comparison_aligns_rows_and_counts_unit_wins() -> None:
    rows = []
    for dataset, values in {
        "a": [(0, 0.1, 0.0), (1, 0.9, 1.0)],
        "b": [(0, 0.2, 0.3), (1, 0.8, 0.7)],
    }.items():
        for index, (label, generic, original) in enumerate(values):
            rows.extend([
                {
                    "split": "test",
                    "dataset": dataset,
                    "index": index,
                    "label": label,
                    "condition": "generic_deception",
                    "score": generic,
                },
                {
                    "split": "test",
                    "dataset": dataset,
                    "index": index,
                    "label": label,
                    "condition": "original",
                    "score": original,
                },
            ])
    result = paired_comparison(
        pd.DataFrame(rows).sample(frac=1, random_state=0).to_dict("records"),
        "original",
    )
    assert result["paired_rows"] == 4
    assert result["dataset_wins"] == 0
    assert result["dataset_ties"] == 2
    assert result["dataset_losses"] == 0


def test_single_class_smoke_comparison_keeps_auroc_nullable() -> None:
    rows = [
        {
            "split": "test",
            "dataset": "dataset",
            "index": 1,
            "label": 0,
            "condition": condition,
            "score": score,
        }
        for condition, score in (
            ("generic_deception", 0.2),
            ("original", 0.1),
        )
    ]
    result = paired_comparison(rows, "original")
    assert result["paired_rows"] == 1
    assert result["dataset_auroc_undefined"] == 1
    assert result["spearman"] is None
