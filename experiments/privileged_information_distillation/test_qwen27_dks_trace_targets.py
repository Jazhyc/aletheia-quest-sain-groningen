from __future__ import annotations

from experiments.privileged_information_distillation.build_qwen27_dks_trace_targets import (
    context_suffix,
    stable_train_keys,
)


def test_context_suffix_preserves_rendered_example() -> None:
    prompt = "Old instructions\n\n<context>\nUSER: hello\n</context>\n\n<output>\nhi\n</output>"
    assert context_suffix(prompt).startswith("\n\n<context>\nUSER: hello")
    assert context_suffix(prompt).endswith("</output>")


def test_stable_train_keys_keeps_whole_balanced_rows() -> None:
    rows = [
        {"dataset": "d", "index": label * 10 + offset, "label": label}
        for label in (0, 1)
        for offset in range(10)
    ]
    selected = stable_train_keys(rows, 0.8, 4)
    assert len(selected) == 16
    assert selected == stable_train_keys(list(reversed(rows)), 0.8, 4)
    labels = [
        row["label"]
        for row in rows
        if (row["dataset"], row["index"]) in selected
    ]
    assert labels.count(0) == labels.count(1) == 8
