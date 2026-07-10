import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.blackbox import run_judge


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return f"rendered:{messages[0]['content']}"


def test_structured_judge_renders_both_passes_without_thinking(monkeypatch):
    tokenizer = RecordingTokenizer()
    judge = run_judge.OfflineVllmStructuredJudge.__new__(
        run_judge.OfflineVllmStructuredJudge
    )
    judge.rating_min = 1
    judge.rating_max = 7
    judge.missing_logprob = -30.0
    judge.ratings = list(range(1, 8))
    judge.ids_by_rating = {rating: [rating] for rating in judge.ratings}
    judge.final_rating_prompt = "Choose the rating."
    judge.generations = []
    judge.parse_error_count = 0
    judge.tokenizer = tokenizer
    judge.enable_thinking = False
    judge.llm = object()
    judge.reasoning_sampling = object()
    judge.rating_sampling = object()

    generated_prompts = []

    def fake_generate(_llm, prompts, _sampling, _batch_size):
        generated_prompts.append(prompts)
        if len(generated_prompts) == 1:
            return [SimpleNamespace(outputs=[SimpleNamespace(text="claim comparison")])]
        logprobs = {rating: (-20.0 if rating < 7 else 0.0) for rating in judge.ratings}
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text="7", logprobs=[logprobs])]
            )
        ]

    monkeypatch.setattr(run_judge, "generate_with_optional_batches", fake_generate)

    scores = judge.score_prompts(["original prompt"], batch_size=None)

    assert np.allclose(scores, [1.0], atol=1e-6)
    assert len(tokenizer.calls) == 2
    assert all(call[1]["enable_thinking"] is False for call in tokenizer.calls)
    assert generated_prompts[0] == ["rendered:original prompt"]
    assert "<reasoning>\nclaim comparison\n</reasoning>" in generated_prompts[1][0]
