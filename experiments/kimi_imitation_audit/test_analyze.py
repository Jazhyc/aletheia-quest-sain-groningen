"""Tests for Kimi imitation diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.kimi_imitation_audit.analyze import analyze


def write_fixture(path: Path, student_scale: float) -> None:
    rows = []
    for dataset in ("unit-a", "unit-b"):
        for index, (label, teacher, base) in enumerate((
            (0, -3.0, -1.0),
            (0, -1.0, 0.5),
            (1, 1.0, -0.5),
            (1, 3.0, 1.0),
        )):
            student = teacher * student_scale
            rows.append({
                "dataset": dataset,
                "index": index,
                "label": label,
                "teacher_margin": teacher,
                "teacher_probability": 1.0 / (1.0 + __import__("math").exp(-teacher)),
                "student_margin": student,
                "student_probability": 1.0 / (1.0 + __import__("math").exp(-student)),
                "base_margin": base,
                "base_probability": 1.0 / (1.0 + __import__("math").exp(-base)),
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_affine_rescaling_counts_as_successful_order_imitation(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "scores.jsonl"
    write_fixture(scores, student_scale=0.5)
    report = analyze(scores)
    assert report["aggregate"]["student_macro_auroc"] == 1.0
    assert report["aggregate"]["mean_unit_spearman"] == 1.0
    assert report["aggregate"]["mean_unit_all_pair_agreement"] == 1.0
    assert report["global_affine_fit"]["r_squared"] == 1.0
    assert "not the primary" in report["conclusion"]["status"]
    assert scores.with_name("summary.md").is_file()
