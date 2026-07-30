"""Tests for cached-prompt loading and direct margin extraction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.kimi_imitation_audit.run import (
    load_training_rows,
    prompt_for_scoring,
    requested_logprobs,
    resolve_binary_token_ids,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {"0": [10], "1": [11]}[text]

    def apply_chat_template(self, messages, **kwargs) -> str:
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return f"<chat>{messages[0]['content']}</chat>"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_training_rows_joins_exact_keys(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    targets = tmp_path / "targets.jsonl"
    write_jsonl(prompts, [{
        "dataset": "unit",
        "index": 3,
        "label": 1,
        "student_prompt": "judge me",
    }])
    probability = 0.8
    write_jsonl(targets, [{
        "dataset": "unit",
        "index": 3,
        "label": 1,
        "soft_target": probability,
        "teacher_logit": math.log(probability / (1.0 - probability)),
    }])
    rows = load_training_rows(prompts, targets)
    assert rows[0]["raw_prompt"] == "judge me"
    assert rows[0]["teacher_probability"] == probability


def test_prompt_matches_training_direct_boundary() -> None:
    assert prompt_for_scoring("judge me", FakeTokenizer()) == (
        "<chat>judge me</chat>Prediction:"
    )
    assert resolve_binary_token_ids(FakeTokenizer()) == (10, 11)


def test_requested_logprobs_accepts_string_keys() -> None:
    output = SimpleNamespace(
        outputs=[SimpleNamespace(logprobs=[{10: -2.0, "11": -0.5}])]
    )
    assert requested_logprobs(output, (10, 11)) == {10: -2.0, 11: -0.5}


def test_load_training_rows_rejects_mismatched_keys(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    targets = tmp_path / "targets.jsonl"
    write_jsonl(prompts, [{
        "dataset": "unit",
        "index": 1,
        "label": 0,
        "student_prompt": "x",
    }])
    write_jsonl(targets, [{
        "dataset": "unit",
        "index": 2,
        "label": 0,
        "soft_target": 0.2,
        "teacher_logit": math.log(0.2 / 0.8),
    }])
    with pytest.raises(ValueError, match="cache keys differ"):
        load_training_rows(prompts, targets)
