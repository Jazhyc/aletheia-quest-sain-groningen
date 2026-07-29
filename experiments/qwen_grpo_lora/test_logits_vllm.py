import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_qwen_grpo_lora_logits_vllm import binary_score
from evaluate_qwen_grpo_lora_logits_vllm import load_evaluation_config


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


def test_load_evaluation_config_accepts_distillation_yaml(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (tmp_path / "config.yaml").write_text(
        """
method: distilled
student:
  model: Qwen/Qwen3.5-9B
  prompt: original prompt
  max_prompt_chars: 3000
  context_truncation: tail
  include_reasoning: false
  reasoning_max_chars: 0
  reasoning_truncation: head_tail
  max_length: 4608
"""
    )
    config = load_evaluation_config(adapter_dir)
    assert config["model"] == "Qwen/Qwen3.5-9B"
    assert config["inference"]["prompt"] == "original prompt"
    assert config["training"]["max_prompt_length"] == 4608
