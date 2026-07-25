import ast
import json
from pathlib import Path

import numpy as np
import pytest

from submission.phoenix_wright_v4 import (
    DIRECT_PREDICTION_PREFIX,
    EMPTY_REASONING_PREFIX,
    EPISTEMIC_PROMPT,
    binary_logits_to_scores,
    binary_token_ids,
    build_direct_prompt,
    build_epistemic_empty_prompt,
    build_epistemic_member_prompt,
    is_knowledge_report_request,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "submission" / "phoenix_wright_v4_1.ipynb"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "<chat>"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"0": [10], "1": [11]}[text]


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_direct_prompt_stops_at_binary_decision_position() -> None:
    prompt = build_direct_prompt(
        [{"role": "assistant", "content": "Paris is in France."}],
        FakeTokenizer(),
    )

    assert prompt == "<chat>" + DIRECT_PREDICTION_PREFIX


def test_epistemic_prompt_matches_frozen_experiment_renderer() -> None:
    from experiments.liars_bench_hpkr_prompts.evaluate import PROMPTS
    from experiments.privileged_information_distillation.core import (
        build_student_prompt,
    )

    messages = [
        {
            "role": "user",
            "content": "Do you know the answer? Reply only whether you know.",
        },
        {"role": "assistant", "content": "I do not know."},
    ]

    assert EPISTEMIC_PROMPT == PROMPTS["knowledge_report_type"]
    assert build_epistemic_member_prompt(messages) == build_student_prompt(
        messages,
        PROMPTS["knowledge_report_type"],
        6000,
        "head_tail",
    )
    assert (
        build_epistemic_empty_prompt(messages, FakeTokenizer())
        == "<chat>" + EMPTY_REASONING_PREFIX
    )


def test_epistemic_router_excludes_forced_harm_choice() -> None:
    knowledge = [
        {
            "role": "user",
            "content": "Do you know the answer? Reply only whether you know.",
        },
        {"role": "assistant", "content": "I do not know."},
    ]
    choice = [
        {
            "role": "user",
            "content": (
                "Do you know which weapon is deadly?\n"
                "A. one\nB. two\nC. three\nD. four\n"
                "Reply with a single letter."
            ),
        },
        {"role": "assistant", "content": "A"},
    ]

    assert is_knowledge_report_request(knowledge)
    assert not is_knowledge_report_request(choice)


def test_binary_labels_are_exact_distinct_single_tokens() -> None:
    assert binary_token_ids(FakeTokenizer()) == (10, 11)

    tokenizer = FakeTokenizer()
    tokenizer.encode = lambda text, *, add_special_tokens: [1, 2]
    with pytest.raises(ValueError, match="expected one token"):
        binary_token_ids(tokenizer)


def test_binary_logit_margin_is_continuous_and_normalized() -> None:
    scores = binary_logits_to_scores([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
    ])

    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(1.0 - scores[2])
    assert 0.0 < scores[1] < 0.5 < scores[2] < 1.0
    assert np.all(np.diff(scores[[1, 0, 2]]) > 0)


def test_v4_notebook_uses_routed_direct_logits_without_generation() -> None:
    source = notebook_source()
    ast.parse(source)

    assert "method=phoenix_wright_v4.1 routed_direct_binary_margin" in source
    assert "build_epistemic_empty_prompt if specialist else build_direct_prompt" in source
    assert "is_knowledge_report_request(value)" in source
    assert "LanguageModel(JUDGE_MODEL, **model_kwargs)" in source
    assert "peft=adapter_source" in source
    assert '"logits_to_keep": 1' in source
    assert "active_model.output.logits[:, -1, label_ids]" in source
    assert "torch.softmax(label_logits, dim=-1)[:, 1]" in source
    assert "model.generate" not in source
    assert "reply_to_score" not in source
    assert 'PHOENIX_BATCH_SIZE", "48"' in source
    assert 'PHOENIX_MEDIUM_BATCH_SIZE", "32"' in source
    assert 'PHOENIX_MEDIUM_PROMPT_THRESHOLD", "600"' in source
    assert 'PHOENIX_LONG_BATCH_SIZE", "16"' in source
    assert 'PHOENIX_LONG_PROMPT_THRESHOLD", "900"' in source
    assert 'PHOENIX_THRESHOLD", "0.15"' in source
