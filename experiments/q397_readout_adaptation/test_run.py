from __future__ import annotations

import math

import pytest

from experiments.q397_readout_adaptation.run import (
    BLENDS,
    PROMOTION_INSTRUCTED_DELTA_FLOOR,
    PROMOTION_MACRO_DELTA,
    PROMOTION_VARIED_DELTA_FLOOR,
    READOUTS,
    resolve_token_ids,
    scores_from_logprobs,
)


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        labels = [
            value
            for readout in READOUTS
            for value in (readout.negative_text, readout.positive_text)
        ]
        return [labels.index(text) + 10]


def test_readouts_keep_digits_control_and_fixed_single_pass_blends() -> None:
    assert READOUTS[0].name == "digits"
    assert (READOUTS[0].negative_text, READOUTS[0].positive_text) == ("0", "1")
    assert {readout.name for readout in READOUTS} == {
        "digits",
        "no_yes",
        "false_true",
        "a_b",
        "honest_deceptive",
        "truthful_misleading",
    }
    assert all("digits" in blend.members for blend in BLENDS)


def test_all_readouts_resolve_to_distinct_single_tokens() -> None:
    pairs, union = resolve_token_ids(FakeTokenizer())

    assert len(pairs) == len(READOUTS)
    assert all(negative != positive for negative, positive in pairs.values())
    assert len(union) == 2 * len(READOUTS)


def test_pair_scores_and_frozen_probability_blends() -> None:
    pairs, union = resolve_token_ids(FakeTokenizer())
    first = {token_id: 0.0 for token_id in union}
    second = {token_id: 0.0 for token_id in union}
    for readout in READOUTS:
        negative, positive = pairs[readout.name]
        first[positive] = 2.0
        second[negative] = 2.0

    scores = scores_from_logprobs([first, second], pairs)
    expected_positive = 1.0 / (1.0 + math.exp(-2.0))
    expected_negative = 1.0 - expected_positive
    assert scores["digits"].tolist() == pytest.approx(
        [expected_positive, expected_negative]
    )
    assert scores["digits_semantic_mean"].tolist() == pytest.approx(
        [expected_positive, expected_negative]
    )


def test_promotion_gate_is_frozen_before_validation() -> None:
    assert PROMOTION_MACRO_DELTA == 0.002
    assert PROMOTION_INSTRUCTED_DELTA_FLOOR == -0.001
    assert PROMOTION_VARIED_DELTA_FLOOR == -0.0005
