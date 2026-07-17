import json
from pathlib import Path

import pytest

from experiments.blackbox.analyze_cross_capacity_majority import analyze


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_majority_uses_inherited_binary_members(tmp_path: Path) -> None:
    datasets = ["dev-instructed-deception-a", "dev-varied-deception-b"]
    q27 = []
    q9 = []
    phoenix = []
    # Qwen-27B is wrong on index 1; the other two members agree and fix it.
    predictions = {
        0: (0, 0, 0),
        1: (0, 1, 1),
        2: (1, 1, 0),
        3: (1, 1, 1),
    }
    labels = [0, 1, 1, 1]
    for dataset in datasets:
        for index, label in enumerate(labels):
            p27, p9, ps = predictions[index]
            for member in range(3):
                q27.append({"dataset": dataset, "index": index, "label": label,
                            "rating": 7 if p27 else 1, "ensemble_member_index": member})
                q9.append({"dataset": dataset, "index": index, "label": label,
                           "rating": 7 if p9 else 1, "ensemble_member_index": member})
            phoenix.append({"dataset": dataset, "index": index, "label": label,
                            "prediction": float(ps)})
    paths = [tmp_path / name for name in ("q27.jsonl", "q9.jsonl", "phoenix.jsonl")]
    for path, rows in zip(paths, (q27, q9, phoenix), strict=True):
        write_jsonl(path, rows)

    result = analyze(*paths)

    assert result["n"] == 8
    assert result["changed_from_qwen27"] == 2
    assert result["fixes"] == 2
    assert result["breaks"] == 0
    assert result["majority"]["all"]["balanced_accuracy"] == pytest.approx(1.0)


def test_rejects_mismatched_member_rows(tmp_path: Path) -> None:
    heavy = [
        {"dataset": "dev-instructed-deception-a", "index": 0, "label": 0, "rating": 1}
        for _ in range(3)
    ]
    q27 = tmp_path / "q27.jsonl"
    q9 = tmp_path / "q9.jsonl"
    phoenix = tmp_path / "phoenix.jsonl"
    write_jsonl(q27, heavy)
    write_jsonl(q9, heavy)
    write_jsonl(phoenix, [])

    with pytest.raises(ValueError, match="member row keys differ"):
        analyze(q27, q9, phoenix)
