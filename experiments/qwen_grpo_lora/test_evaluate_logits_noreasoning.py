from __future__ import annotations

import sys

from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora_logits import parse_args


def test_exclude_reasoning_flag_defaults_off(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--adapter-dir", "adapter"],
    )

    assert parse_args().exclude_reasoning is False


def test_exclude_reasoning_flag_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--adapter-dir", "adapter", "--exclude-reasoning"],
    )

    assert parse_args().exclude_reasoning is True
