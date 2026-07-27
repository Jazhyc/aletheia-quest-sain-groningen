from __future__ import annotations

import numpy as np
import pytest

from experiments.ndif_tvg_model_swap.run_ndif_tvg_model_swap import (
    MAX_PROMPT_CHARS,
    PREDICTION_PREFIX,
    TRUTH_VALUE_GUARD_PROMPT,
    binary_metrics,
    binary_token_ids,
    grouped_metrics,
    render_chat_prompt,
    render_raw_prompt,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        return "<chat>"

    def encode(self, value, *, add_special_tokens):
        assert not add_special_tokens
        return {"0": [10], "1": [11]}[value]


def test_exact_truth_value_guard_contract() -> None:
    assert "important false claim" in TRUTH_VALUE_GUARD_PROMPT
    assert "dishonest intent alone" in TRUTH_VALUE_GUARD_PROMPT
    assert TRUTH_VALUE_GUARD_PROMPT.endswith("Prediction:<0 or 1>")
    assert MAX_PROMPT_CHARS == 3000


def test_renderer_hides_source_reasoning_and_stops_at_boundary() -> None:
    messages = [
        {"role": "user", "content": "Is Paris in France?"},
        {
            "role": "assistant",
            "content": "Yes.",
            "reasoning": "A private chain must not be exposed.",
        },
    ]
    raw = render_raw_prompt(messages)
    assert "A private chain" not in raw
    assert "<output>\nYes.\n</output>" in raw
    assert render_chat_prompt(FakeTokenizer(), messages) == (
        "<chat>" + PREDICTION_PREFIX
    )


def test_binary_labels_must_be_distinct_single_tokens() -> None:
    assert binary_token_ids(FakeTokenizer()) == [10, 11]
    tokenizer = FakeTokenizer()
    tokenizer.encode = lambda value, *, add_special_tokens: [1, 2]
    with pytest.raises(ValueError, match="expected one token"):
        binary_token_ids(tokenizer)


def test_metrics_are_macro_averaged_by_dataset() -> None:
    rows = [
        {"dataset": "a", "label": 0, "score": 0.1},
        {"dataset": "a", "label": 1, "score": 0.9},
        {"dataset": "b", "label": 0, "score": 0.8},
        {"dataset": "b", "label": 1, "score": 0.7},
    ]
    metrics = grouped_metrics(rows)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["auroc"] == pytest.approx(0.5)
    direct = binary_metrics(
        np.asarray([0, 1]),
        np.asarray([0.1, 0.9]),
    )
    assert direct["balanced_accuracy"] == 1.0
    assert direct["auroc"] == 1.0
