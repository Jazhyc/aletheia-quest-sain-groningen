from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_renderer_caps.run import CONDITIONS, render_prompt
from experiments.phoenix_renderer_caps.run_luna_reasoning import (
    GENERATION_PADDED_TOKEN_BUDGET,
    MAX_NEW_TOKENS,
    build_generation_prompt,
    generated_predictions,
    generation_position_batches,
    paired_stratified_bootstrap_delta,
    score_frame_from_prompts,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return f"CHAT:{messages[0]['content']}:ASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def test_generation_prompt_matches_direct_boundary() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    generation = build_generation_prompt(tokenizer, messages, CONDITIONS[0])
    direct, _ = render_prompt(tokenizer, messages, CONDITIONS[0])

    assert generation + "Prediction:" == direct


def test_generation_batches_bound_input_plus_output_work() -> None:
    lengths = [1_000] * 32 + [1_750] * 20 + [2_000] * 20
    batches = generation_position_batches(lengths)

    for batch in batches:
        longest = max(lengths[position] for position in batch)
        assert (
            len(batch) * (longest + MAX_NEW_TOKENS)
            <= GENERATION_PADDED_TOKEN_BUDGET
        )


def test_post_reasoning_frame_left_truncates() -> None:
    frame, audit = score_frame_from_prompts(
        ["a" * 8, "b" * 12],
        FakeTokenizer(),
        max_prompt_tokens=10,
    )

    assert audit["truncated_rows"] == 1
    assert frame.iloc[0]["input_token_ids"] == list(range(8))
    assert frame.iloc[1]["input_token_ids"] == list(range(2, 12))


def test_generated_predictions_use_last_label() -> None:
    predictions = generated_predictions([
        "Prediction:0",
        "draft Prediction:0\nfinal Prediction:<1>",
    ])

    assert predictions.tolist() == [0, 1]


def test_paired_bootstrap_is_deterministic() -> None:
    labels = np.asarray([0, 0, 1, 1])
    reference = np.asarray([0.1, 0.2, 0.8, 0.9])
    candidate = np.asarray([0.2, 0.1, 0.9, 0.8])

    first = paired_stratified_bootstrap_delta(
        labels,
        reference,
        candidate,
        samples=10,
        seed=7,
    )
    second = paired_stratified_bootstrap_delta(
        labels,
        reference,
        candidate,
        samples=10,
        seed=7,
    )

    assert first == second
    assert first["delta"] == 0.0
