import pandas as pd

from experiments.wikidata_rag.evaluate_matched_reader import (
    changed_prompt_mask,
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
