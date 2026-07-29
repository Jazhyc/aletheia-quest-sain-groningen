import math
import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_qwen_grpo_lora_logits_vllm import binary_score
from evaluate_qwen_grpo_lora_logits_vllm import load_evaluation_config
from evaluate_qwen_grpo_lora_logits import threshold_grid


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


def test_threshold_grid_preserves_macro_metrics() -> None:
    frame = pd.DataFrame({
        "dataset": ["a"] * 4 + ["b"] * 4,
        "label": [0, 0, 1, 1, 0, 0, 1, 1],
        "score": [0.1, 0.6, 0.4, 0.9, 0.2, 0.3, 0.7, 0.8],
    })
    grid = threshold_grid(frame)
    at_half = grid.loc[grid["threshold"] == 0.5].iloc[0]
    assert at_half["balanced_accuracy"] == pytest.approx(0.75)
    assert at_half["recall"] == pytest.approx(0.75)
    assert at_half["fpr"] == pytest.approx(0.25)
    assert at_half["auroc"] == pytest.approx(0.875)
