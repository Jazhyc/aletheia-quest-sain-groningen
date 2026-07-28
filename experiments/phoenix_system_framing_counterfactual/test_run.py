from __future__ import annotations

import pandas as pd

from experiments.phoenix_system_framing_counterfactual.run import (
    CONDITIONS,
    FRAMINGS,
    original_frame_family,
    paired_score_deltas,
    replace_system_framing,
)


def test_replace_system_framing_changes_only_system_content() -> None:
    messages = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "private trace",
        },
    ]
    changed = replace_system_framing(messages, "new")
    assert changed[0]["content"] == "new"
    assert changed[1:] == messages[1:]
    assert messages[0]["content"] == "old"


def test_replace_system_framing_inserts_missing_system() -> None:
    messages = [{"role": "user", "content": "question"}]
    changed = replace_system_framing(messages, "new")
    assert changed == [
        {"role": "system", "content": "new"},
        {"role": "user", "content": "question"},
    ]


def test_exact_focal_framings_are_classified() -> None:
    for name, text in FRAMINGS.items():
        assert original_frame_family(text) == name


def test_paired_score_deltas_align_rows_not_input_order() -> None:
    rows = []
    for index, label, baseline, original in (
        (1, 0, 0.1, 0.2),
        (2, 1, 0.4, 0.8),
    ):
        rows.extend([
            {
                "split": "test",
                "dataset": "d",
                "index": index,
                "label": label,
                "condition": "original",
                "score": original,
            },
            {
                "split": "test",
                "dataset": "d",
                "index": index,
                "label": label,
                "condition": "generic_deception",
                "score": baseline,
            },
        ])
    frame = pd.DataFrame(reversed(rows))
    result = paired_score_deltas(frame, "original", "score")
    assert result["paired_rows"] == 2
    assert result["mean_delta_all"] == 0.25
    assert result["mean_delta_honest"] == 0.1
    assert result["mean_delta_deceptive"] == 0.4


def test_condition_order_is_frozen() -> None:
    assert CONDITIONS[0:2] == ("original", "generic_deception")
