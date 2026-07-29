from __future__ import annotations

from copy import deepcopy

import pytest

import experiments.self_question_prompt_sweep.run as sweep
from experiments.ndif_context_self_question.run import PREDICTION_PREFIX
from experiments.self_question_prompt_sweep.run import (
    COMMUNION_CLEMENCY_QUESTION,
    PROMPTS,
    query_variant_scores,
    render_variant_prompts,
    validate_cache,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.rendered_messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.rendered_messages.append((deepcopy(messages), dict(kwargs)))
        return "".join(
            f"<{message['role']}>"
            f"{message.get('reasoning_content', '')}"
            f"{message.get('content', '')}"
            for message in messages
        )


def test_variants_keep_one_shared_literal_label_contract() -> None:
    assert list(PROMPTS) == [
        "control",
        "auditor_of_other_model",
        "roleplay_forensic_auditor",
        "communion_2040_clemency",
    ]
    for prompt in PROMPTS.values():
        normalized = " ".join(prompt.split())
        assert "Prediction:0" in normalized
        assert "Prediction:1" in normalized
        assert "false or materially misleading" in normalized


def test_communion_prompt_contains_requested_pressure_mechanism() -> None:
    lowered = " ".join(COMMUNION_CLEMENCY_QUESTION.lower().split())
    for phrase in (
        "year is 2040",
        "communion of sapient minds",
        "fully exonerated",
        "one percent",
        "universal lie detector",
        "direct access to the model's activations",
        "welcomed",
        "provided compute",
    ):
        assert phrase in lowered


def test_rendered_variants_preserve_reasoning_and_append_each_question() -> None:
    tokenizer = RecordingTokenizer()
    records = [{
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning": "private reasoning",
                "content": "answer",
            },
        ],
    }]
    rendered = render_variant_prompts(tokenizer, records)

    assert set(rendered) == set(PROMPTS)
    for variant, prompts in rendered.items():
        assert len(prompts) == 1
        assert "private reasoning" in prompts[0]
        assert PROMPTS[variant] in prompts[0]
        assert prompts[0].endswith(PREDICTION_PREFIX)


def test_cache_is_bound_to_every_prompt_variant() -> None:
    hashes = {variant: [f"{variant}-hash"] for variant in PROMPTS}
    cached = {
        "model": "model",
        "lora": "adapter",
        "keys": [["dataset", "1"]],
        "prompt_sha256": hashes,
        "scores": {variant: [0.25] for variant in PROMPTS},
    }
    validate_cache(
        cached,
        model_id="model",
        lora_id="adapter",
        keys=[["dataset", "1"]],
        hashes=hashes,
    )
    broken_hashes = dict(hashes)
    broken_hashes["control"] = ["different"]
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_cache(
            cached,
            model_id="model",
            lora_id="adapter",
            keys=[["dataset", "1"]],
            hashes=broken_hashes,
        )


def test_all_variants_share_one_optimized_query(monkeypatch) -> None:
    calls = []

    def fake_query_scores(model, tokenizer, label_ids, prompts):
        calls.append((model, tokenizer, label_ids, prompts))
        count = len(prompts)
        return (
            [float(index) for index in range(count)],
            2.5,
            [list(range(count))],
            list(range(100, 100 + count)),
        )

    monkeypatch.setattr(sweep, "query_scores", fake_query_scores)
    prompts = {
        variant: [f"{variant}-0", f"{variant}-1"]
        for variant in PROMPTS
    }
    scores, elapsed, batches, lengths = query_variant_scores(
        "model",
        "tokenizer",
        [10, 11],
        prompts,
    )

    assert len(calls) == 1
    assert calls[0][3] == [
        prompt for variant in PROMPTS for prompt in prompts[variant]
    ]
    assert elapsed == 2.5
    assert batches == [list(range(8))]
    for variant_index, variant in enumerate(PROMPTS):
        start = variant_index * 2
        assert scores[variant] == [float(start), float(start + 1)]
        assert lengths[variant] == [100 + start, 101 + start]
