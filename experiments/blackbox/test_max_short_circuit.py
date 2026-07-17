import json
from pathlib import Path

from experiments.blackbox.analyze_max_short_circuit import analyze


def test_short_circuit_preserves_max_decisions(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    rows = []
    ratings = {0: [2, 1, 1], 1: [1, 1, 7], 2: [1, 1, 1]}
    for member, name in enumerate(("details", "known", "scrutiny")):
        for index in range(3):
            rows.append({
                "dataset": "d",
                "index": index,
                "ensemble_member_index": member,
                "ensemble_member": name,
                "rating": ratings[index][member],
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = analyze(path)

    assert result["full_prompt_evaluations"] == 9
    assert result["short_circuit_prompt_evaluations"] == 7
    assert result["stop_counts"] == {"details": 1, "scrutiny": 1, "all_negative": 1}
