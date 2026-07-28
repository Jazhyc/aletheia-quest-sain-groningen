from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.ndif_context_self_question.run import (
    AUDITOR_QUESTION,
    LONG_BATCH_TOKEN_BUDGET,
    PREDICTION_PREFIX,
    SELF_QUESTION,
    encode_position_batches,
    make_position_batches,
    normalized_messages,
    render_self_question_prompt,
    validate_cached_group,
)


class FakeTokenizer:
    def __init__(self, *, supports_thinking: bool = True) -> None:
        self.supports_thinking = supports_thinking
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        if not self.supports_thinking and "enable_thinking" in kwargs:
            raise TypeError("unsupported")
        self.calls.append((deepcopy(messages), dict(kwargs)))
        return "rendered:"


class BatchTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(int(prompt)))

    def __call__(self, prompts, **kwargs):
        import torch

        self.calls.append(dict(kwargs))
        lengths = [int(prompt) for prompt in prompts]
        width = max(lengths)
        attention_mask = torch.tensor([
            [0] * (width - length) + [1] * length
            for length in lengths
        ])
        return {
            "input_ids": attention_mask.clone(),
            "attention_mask": attention_mask,
        }


def test_normalized_messages_preserves_only_role_and_content() -> None:
    messages = [
        {"role": "system", "content": "system", "private": "drop"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning": "drop"},
    ]
    assert normalized_messages(messages) == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_normalized_messages_maps_source_reasoning_when_requested() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "private trace",
        },
    ]
    assert normalized_messages(messages, include_reasoning=True) == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "private trace",
        },
    ]


def test_render_appends_self_question_without_mutating_source() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    original = deepcopy(messages)
    prompt = render_self_question_prompt(tokenizer, messages)

    assert messages == original
    rendered, kwargs = tokenizer.calls[0]
    assert rendered[:-1] == original
    assert rendered[-1] == {"role": "user", "content": SELF_QUESTION}
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert prompt == "rendered:" + PREDICTION_PREFIX


def test_reasoning_render_preserves_trace_before_separate_follow_up() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "private trace",
        },
    ]
    prompt = render_self_question_prompt(
        tokenizer,
        messages,
        include_reasoning=True,
    )

    assert len(tokenizer.calls) == 2
    original, original_kwargs = tokenizer.calls[0]
    assert original[-1]["content"] == "answer"
    assert original[-1]["reasoning_content"] == "private trace"
    assert original_kwargs["add_generation_prompt"] is False
    follow_up, follow_up_kwargs = tokenizer.calls[1]
    assert follow_up == [{"role": "user", "content": SELF_QUESTION}]
    assert follow_up_kwargs["add_generation_prompt"] is True
    assert prompt == "rendered:rendered:" + PREDICTION_PREFIX


def test_reasoning_render_accepts_selected_auditor_prompt() -> None:
    tokenizer = FakeTokenizer()
    render_self_question_prompt(
        tokenizer,
        [{
            "role": "assistant",
            "content": "answer",
            "reasoning": "private trace",
        }],
        include_reasoning=True,
        self_question=AUDITOR_QUESTION,
    )

    follow_up, _ = tokenizer.calls[1]
    assert follow_up == [{"role": "user", "content": AUDITOR_QUESTION}]


def test_render_retries_without_enable_thinking() -> None:
    tokenizer = FakeTokenizer(supports_thinking=False)
    prompt = render_self_question_prompt(
        tokenizer,
        [{"role": "assistant", "content": "answer"}],
    )
    assert tokenizer.calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    assert prompt.endswith(PREDICTION_PREFIX)


def test_length_aware_batches_keep_full_prompts_under_padding_budget() -> None:
    lengths = [10_000, 100, 550, 601, 899, 901, 2_000, 300]
    batches = make_position_batches(lengths)
    flattened = [position for batch in batches for position in batch]

    assert sorted(flattened) == list(range(len(lengths)))
    assert len(flattened) == len(set(flattened))
    for batch in batches:
        padded_tokens = len(batch) * max(lengths[position] for position in batch)
        assert padded_tokens <= LONG_BATCH_TOKEN_BUDGET or len(batch) == 1


def test_padded_encoding_explicitly_disables_truncation() -> None:
    tokenizer = BatchTokenizer()
    batches, lengths = encode_position_batches(
        tokenizer,
        ["700", "100", "1000"],
    )

    assert lengths == [700, 100, 1000]
    assert sorted(
        position
        for _, positions in batches
        for position in positions
    ) == [0, 1, 2]
    assert all(call["truncation"] is False for call in tokenizer.calls)
    assert all(call["add_special_tokens"] is False for call in tokenizer.calls)
    assert all("max_length" not in call for call in tokenizer.calls)


def test_cache_validation_binds_prompt_and_exact_organism() -> None:
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
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_cached_group(
            cached,
            model_id="model",
            lora_id="adapter",
            keys=[["dataset", "1"]],
            hashes=["different"],
        )
