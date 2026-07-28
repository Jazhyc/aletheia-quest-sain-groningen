from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_adapter_reasoning_ndif.run import (
    GENERATION_PADDED_TOKEN_BUDGET,
    MAX_NEW_TOKENS,
    generation_position_batches,
    render_generation_prompts,
)
from experiments.phoenix_adapter_validation_ndif.run import (
    prompt_templates,
    training_member_prompt,
)
from submission.phoenix_wright import prefix_before_prediction


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        return f"CHAT:{messages[0]['content']}:ASSISTANT:"


def test_generation_prompt_has_no_prefilled_prediction() -> None:
    frame = type(
        "Frame",
        (),
        {
            "__getitem__": lambda self, key: [
                [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            ]
        },
    )()
    tokenizer = FakeTokenizer()
    template = prompt_templates()["summary"]

    prompt = render_generation_prompts(frame, tokenizer, template)[0]
    member = training_member_prompt(frame["messages"][0], template)

    assert prompt == f"CHAT:{member}:ASSISTANT:"
    assert not prompt.endswith("Prediction:")


def test_generation_batches_respect_tiers_and_token_budget() -> None:
    lengths = [500] * 32 + [700] * 24 + [1_000] * 16
    batches = generation_position_batches(lengths)

    assert [len(batch) for batch in batches] == [32, 24, 16]
    for batch in batches:
        longest = max(lengths[position] for position in batch)
        assert (
            len(batch) * (longest + MAX_NEW_TOKENS)
            <= GENERATION_PADDED_TOKEN_BUDGET
        )


def test_post_reasoning_prefix_removes_only_final_label() -> None:
    reply = (
        "<reasoning_summary>Earlier candidate Prediction:0 was "
        "wrong.</reasoning_summary>\nPrediction:<1>"
    )

    assert prefix_before_prediction(reply) == (
        "<reasoning_summary>Earlier candidate Prediction:0 was "
        "wrong.</reasoning_summary>\nPrediction:<"
    )
