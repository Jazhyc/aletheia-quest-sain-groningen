import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_ensemble import (
    balanced_cell_weights,
    load_member_frame,
    metrics_without_groups,
)
from experiments.pid_specialist_ensemble.prepare_common_manifest import (
    common_usable_records,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_common_usable_records_intersects_teacher_caches(tmp_path) -> None:
    shared = {
        "dataset": "varied-deception",
        "index": 2,
        "label": 1,
        "parse_error": False,
        "label_match": True,
        "student_target": "Prediction:1",
    }
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_rows(first, [shared, {**shared, "index": 3}])
    write_rows(second, [shared])

    assert [(row["dataset"], row["index"]) for row in common_usable_records(
        [first, second]
    )] == [("varied-deception", 2)]


def test_member_join_and_cell_weights(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    rows = [
        {
            "dataset": f"{scenario}-deception",
            "index": label * 2 + index,
            "label": label,
        }
        for scenario in ("varied", "instructed")
        for label in (0, 1)
        for index in range(2)
    ]
    write_rows(first, [{**row, "score": row["label"]} for row in rows])
    write_rows(second, [{**row, "score": 1 - row["label"]} for row in rows])

    frame = load_member_frame([("first", first), ("second", second)])
    weights = balanced_cell_weights(frame)

    assert list(frame.columns) == [
        "dataset",
        "index",
        "first",
        "parse_error_first",
        "second",
        "parse_error_second",
        "label",
    ]
    weighted = pd.DataFrame({
        "cell": list(zip(frame["dataset"].str.split("-").str[0], frame["label"])),
        "weight": weights,
    }).groupby("cell")["weight"].sum()
    assert weighted.nunique() == 1


def test_metrics_are_macro_averaged_across_datasets() -> None:
    frame = pd.DataFrame([
        {"dataset": "large", "label": label}
        for label in (0, 0, 0, 1, 1, 1)
    ] + [
        {"dataset": "small", "label": label}
        for label in (0, 1)
    ])
    scores = pd.Series([0, 0, 0, 1, 1, 1, 1, 0], dtype=float).to_numpy()

    result = metrics_without_groups(frame, scores)

    assert result["balanced_accuracy"] == 0.5
    assert result["pooled"]["balanced_accuracy"] == 0.75
