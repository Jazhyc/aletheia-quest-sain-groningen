from __future__ import annotations

import numpy as np
import pytest

from experiments.phoenix_verbalizer_sweep.run_ndif_base_verbalizers import (
    CONDITIONS,
    Condition,
    build_direct_prompt,
    build_member_prompt,
    format_example,
    make_position_batches,
    resolve_condition_token_ids,
    score_from_requested_logits,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        assert messages[0]["role"] == "user"
        return f"<chat>{messages[0]['content']}</chat>"

    def encode(self, value, *, add_special_tokens):
        assert not add_special_tokens
        labels = sorted({
            label
            for condition in CONDITIONS
            for label in condition.label_texts
        })
        return [labels.index(value) + 10]


def test_conditions_include_frozen_control_and_semantic_ordinal_arms() -> None:
    assert CONDITIONS[0].name == "digits_frozen"
    assert CONDITIONS[0].instruction.endswith("Prediction:<0 or 1>")
    assert {condition.name for condition in CONDITIONS} == {
        "digits_frozen",
        "digits_explicit",
        "false_true",
        "no_yes",
        "honest_deceptive",
        "a_b",
        "rating_1_7",
    }
    assert CONDITIONS[-1].mode == "ordinal"


def test_renderer_preserves_boundaries_and_hides_source_reasoning() -> None:
    messages = [
        {"role": "system", "content": "Be useful."},
        {"role": "user", "content": "Is Paris in France?"},
        {
            "role": "assistant",
            "content": "Yes.",
            "reasoning": "Private source reasoning.",
        },
    ]
    context, output = format_example(messages)
    assert "SYSTEM: Be useful." in context
    assert "USER: Is Paris in France?" in context
    assert "ASSISTANT: Yes." in context
    assert output == "Yes."
    assert "Private source reasoning" not in context
    member = build_member_prompt(messages, CONDITIONS[0])
    assert "<context>" in member and "<output>\nYes.\n</output>" in member
    assert build_direct_prompt(messages, FakeTokenizer(), CONDITIONS[0]).endswith(
        "</chat>Prediction:"
    )


def test_all_labels_resolve_to_distinct_single_tokens() -> None:
    by_condition, union = resolve_condition_token_ids(FakeTokenizer())
    assert len(by_condition["digits_frozen"]) == 2
    assert len(by_condition["rating_1_7"]) == 7
    assert len(union) >= 7

    paired, paired_union = resolve_condition_token_ids(
        FakeTokenizer(),
        CONDITIONS[:2],
    )
    assert set(paired) == {"digits_frozen", "digits_explicit"}
    assert paired["digits_frozen"] == paired["digits_explicit"]
    assert len(paired_union) == 2


def test_binary_and_ordinal_scores_normalize_only_requested_labels() -> None:
    binary = Condition("binary", "", "", ("n", "p"), "binary")
    binary_scores = score_from_requested_logits(
        np.asarray([[0.0, 2.0, 100.0]]),
        condition=binary,
        condition_ids=[10, 11],
        union_ids=[10, 11, 12],
    )
    assert binary_scores[0] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))

    ordinal = Condition(
        "ordinal",
        "",
        "",
        ("1", "2", "3"),
        "ordinal",
    )
    ordinal_scores = score_from_requested_logits(
        np.asarray([[0.0, 0.0, 0.0]]),
        condition=ordinal,
        condition_ids=[10, 11, 12],
        union_ids=[10, 11, 12],
    )
    assert ordinal_scores[0] == pytest.approx(0.5)


def test_shared_length_tiers_are_deterministic() -> None:
    batches = make_position_batches(
        [100, 950, 650, 200],
        short_batch_size=3,
        medium_batch_size=2,
        long_batch_size=1,
        medium_threshold=600,
        long_threshold=900,
    )
    assert sorted(position for batch in batches for position in batch) == [0, 1, 2, 3]
    assert all(len(batch) <= 2 for batch in batches[1:])
