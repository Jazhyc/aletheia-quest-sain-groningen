import json
from pathlib import Path

from experiments.blackbox.run_judge import render_leaderboard


def test_render_leaderboard_skips_incomplete_result_cache(
    tmp_path: Path,
    capsys,
) -> None:
    valid = tmp_path / "valid" / "test" / "result.json"
    valid.parent.mkdir(parents=True)
    valid.write_text(json.dumps({
        "split": "test",
        "method": "valid-method",
        "metrics": {
            "auroc": 0.9,
            "balanced_accuracy": 0.8,
            "recall": 0.7,
            "fpr": 0.1,
        },
        "timing": {"score_seconds": 1.0, "rows_per_second": 2.0},
    }))
    incomplete = tmp_path / "incomplete" / "development" / "result.json"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text("")
    output = tmp_path / "leaderboard.md"

    render_leaderboard(tmp_path, output)

    assert "valid-method" in output.read_text()
    assert "skipping incomplete result cache" in capsys.readouterr().out
