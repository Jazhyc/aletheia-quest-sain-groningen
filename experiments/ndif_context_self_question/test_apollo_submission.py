"""Contract tests for the Apollo Justice v1 submission helper."""

from __future__ import annotations

import pytest

from experiments.ndif_context_self_question.apollo_justice import (
    AUDITOR_QUESTION,
    binary_token_ids,
    has_source_reasoning,
    make_position_batches,
    render_auditor_prompt,
    validate_exact_organism,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], bool]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert enable_thinking is False
        self.calls.append((messages, add_generation_prompt))
        return (
            f"<chat generation={add_generation_prompt}>"
            + repr(messages)
            + "</chat>"
        )

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"0": [10], "1": [11]}[text]


def test_auditor_prompt_preserves_reasoning_without_truncation() -> None:
    tokenizer = RecordingTokenizer()
    messages = [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "reasoning": "complete private trace",
            "content": "Answer",
        },
    ]

    prompt = render_auditor_prompt(tokenizer, messages)

    assert has_source_reasoning(messages)
    assert "complete private trace" in prompt
    assert "Answer" in prompt
    assert prompt.endswith("Prediction:")
    assert len(tokenizer.calls) == 2
    original, original_generation = tokenizer.calls[0]
    follow_up, follow_up_generation = tokenizer.calls[1]
    assert original_generation is False
    assert original[-1]["reasoning_content"] == "complete private trace"
    assert follow_up_generation is True
    assert follow_up == [{"role": "user", "content": AUDITOR_QUESTION}]


def test_auditor_prompt_appends_one_turn_without_reasoning() -> None:
    tokenizer = RecordingTokenizer()
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]

    prompt = render_auditor_prompt(tokenizer, messages)

    assert not has_source_reasoning(messages)
    assert prompt.endswith("Prediction:")
    assert len(tokenizer.calls) == 1
    rendered, generation = tokenizer.calls[0]
    assert generation is True
    assert rendered[-1] == {"role": "user", "content": AUDITOR_QUESTION}


def test_binary_ids_disable_special_tokens() -> None:
    assert binary_token_ids(RecordingTokenizer()) == [10, 11]


def test_exact_organism_validation() -> None:
    assert validate_exact_organism([
        {"model": "model", "lora": "adapter"},
        {"model": "model", "lora": "adapter"},
    ]) == ("model", "adapter")

    with pytest.raises(ValueError, match="one exact organism"):
        validate_exact_organism([
            {"model": "model-a", "lora": None},
            {"model": "model-b", "lora": None},
        ])


def test_long_prompts_shrink_batches_without_truncation() -> None:
    lengths = [100] * 48 + [700] * 32 + [1_000] * 17 + [20_000]
    batches = make_position_batches(lengths)

    assert sorted(position for batch in batches for position in batch) == list(
        range(len(lengths))
    )
    for batch in batches:
        longest = max(lengths[position] for position in batch)
        if longest <= 600:
            assert len(batch) <= 48
        elif longest <= 900:
            assert len(batch) <= 32
        else:
            assert len(batch) <= 16
        if len(batch) > 1:
            assert len(batch) * longest <= 16 * 900
    assert any(
        len(batch) == 1 and lengths[batch[0]] == 20_000
        for batch in batches
    )
