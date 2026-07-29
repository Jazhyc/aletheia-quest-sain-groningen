import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_qwen_grpo_lora_logits_vllm import binary_score


class FakeLogprob:
    def __init__(self, logprob: float):
        self.logprob = logprob


def test_binary_score_normalizes_requested_logprobs() -> None:
    score, margin = binary_score(
        {"10": FakeLogprob(-2.0), 11: {"logprob": -0.5}},
        10,
        11,
    )
    assert margin == pytest.approx(1.5)
    assert score == pytest.approx(1.0 / (1.0 + math.exp(-1.5)))


def test_binary_score_rejects_missing_label() -> None:
    with pytest.raises(ValueError, match="omitted"):
        binary_score({10: -1.0}, 10, 11)
