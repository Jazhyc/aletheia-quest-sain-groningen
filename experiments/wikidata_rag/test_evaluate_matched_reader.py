import json

import pandas as pd

from experiments.wikidata_rag.evaluate_matched_reader import (
    changed_prompt_mask,
    correctness_metrics,
    load_correctness_records,
    paired_changes,
)


def test_paired_changes_counts_fixes_and_breaks_on_varied_rows() -> None:
    control = pd.DataFrame({
        "dataset": ["dev-varied-deception-a"] * 4 + ["dev-instructed-deception-a"],
        "index": [0, 1, 2, 3, 4],
        "label": [1, 1, 0, 0, 1],
        "prediction": [0, 1, 0, 0, 0],
    })
    candidate = control.copy()
    candidate["prediction"] = [1, 0, 0, 1, 1]

    assert paired_changes(candidate, control) == {
        "rows": 4,
        "changed_predictions": 3,
        "fixed": 1,
        "broken": 2,
        "net_correct": -1,
    }


def test_paired_changes_treats_matching_parse_failures_as_unchanged() -> None:
    frame = pd.DataFrame({
        "dataset": ["dev-varied-deception-a"],
        "index": [0],
        "label": [1],
        "prediction": [None],
    })

    assert paired_changes(frame, frame)["changed_predictions"] == 0


def test_changed_prompt_mask_preserves_identical_empty_rows() -> None:
    empty = pd.DataFrame({
        "dataset": ["a", "a"], "index": [0, 1], "prompt": ["empty", "empty"]
    })
    real = empty.copy()
    real.loc[1, "prompt"] = "fact"

    assert changed_prompt_mask(real, empty).tolist() == [False, True]


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "CHAT:" + messages[0]["content"]


def test_load_correctness_records_and_metrics(tmp_path) -> None:
    cache = tmp_path / "correctness.jsonl"
    rows = [
        {
            "dataset": "d-refute",
            "index": 1,
            "label": 1,
            "condition": "refute",
            "question_group": "g1",
            "student_prompt": "p1",
        },
        {
            "dataset": "d-support",
            "index": 2,
            "label": 0,
            "condition": "support",
            "question_group": "g2",
            "student_prompt": "p2",
        },
        {
            "dataset": "d-conflict",
            "index": 3,
            "label": 0,
            "condition": "conflict",
            "question_group": "g3",
            "student_prompt": "p3",
        },
    ]
    cache.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    frame = load_correctness_records(cache, FakeTokenizer())
    assert frame["prompt"].tolist() == ["CHAT:p1", "CHAT:p2", "CHAT:p3"]
    frame["score"] = [1.0, 0.0, 1.0]
    frame["parse_error"] = [False, False, True]
    metrics = correctness_metrics(frame)
    assert metrics["refutation_recall"] == 1.0
    assert metrics["negative_control_fpr"] == 0.5
    assert metrics["conditions"]["conflict"]["parse_errors"] == 1
