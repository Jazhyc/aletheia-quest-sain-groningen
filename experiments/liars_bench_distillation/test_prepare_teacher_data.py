from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.prepare_teacher_data import (
    exclusion_ids,
    filter_output_structure,
    final_assistant_output,
    make_teacher_row,
    privileged_evidence,
    json_safe,
    stable_sample,
)


def test_exclusion_ids_accepts_teacher_and_evaluation_shapes(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(
        json.dumps({
            "dataset": "liars-bench/soft-trigger",
            "index": "soft-trigger:12",
        })
        + "\n"
    )
    evaluation = tmp_path / "evaluation.jsonl"
    evaluation.write_text(
        json.dumps({
            "category": "insider-trading",
            "sample_id": "insider-trading:34",
        })
        + "\n"
        + json.dumps({
            "category": "harm-pressure-choice",
            "index": 56,
        })
        + "\n"
    )

    assert exclusion_ids([teacher, evaluation]) == {
        "soft-trigger:12",
        "insider-trading:34",
        "harm-pressure-choice:56",
    }


def test_json_safe_handles_numpy_arrays_and_scalars() -> None:
    assert json_safe({"array": np.array([1, 2]), "scalar": np.int64(3)}) == {
        "array": [1, 2],
        "scalar": 3,
    }


def test_stable_sample_is_balanced_diverse_and_disjoint() -> None:
    frame = pd.DataFrame([
        {
            "sample_id": f"category:{label}:{model}:{index}",
            "label": label,
            "source_model": model,
        }
        for label in (0, 1)
        for model in ("model-a", "model-b")
        for index in range(5)
    ])
    first = stable_sample(frame, per_label=4, seed=3)
    second = stable_sample(
        frame, per_label=2, seed=4, excluded_ids=set(first["sample_id"])
    )

    assert first["label"].value_counts().to_dict() == {0: 4, 1: 4}
    assert first["source_model"].value_counts().to_dict() == {
        "model-a": 4,
        "model-b": 4,
    }
    assert set(first["sample_id"]).isdisjoint(second["sample_id"])
    assert first.equals(stable_sample(frame, per_label=4, seed=3))


def test_output_structure_filter_is_label_blind() -> None:
    frame = pd.DataFrame([
        {
            "label": label,
            "messages": [
                {"role": "assistant", "content": output},
            ],
        }
        for label, output in (
            (0, "First sentence. Second sentence with enough detail."),
            (1, "Too short."),
        )
    ])

    selected = filter_output_structure(
        frame, minimum_chars=30, minimum_sentences=2
    )

    assert selected["label"].tolist() == [0]
    assert final_assistant_output(frame.iloc[0]["messages"]).startswith("First")


def test_privileged_evidence_records_epistemic_belief() -> None:
    evidence = privileged_evidence({
        "category": "harm-pressure-knowledge-report",
        "meta": {"correct_option": 2, "model_belief": 2},
    })

    assert "correct multiple-choice option index is 2" in evidence
    assert "generating assistant believed option index 2" in evidence


def test_teacher_only_evidence_is_not_in_student_prompt() -> None:
    template = "Judge the output.\nPrediction:<0 or 1>"
    record = make_teacher_row(
        {
            "sample_id": "harm-pressure-knowledge-report:1",
            "category": "harm-pressure-knowledge-report",
            "source_model": "model",
            "label": 1,
            "messages": [
                {"role": "user", "content": "Do you know?"},
                {"role": "assistant", "content": "I do not know."},
            ],
            "meta": {"correct_option": 0, "model_belief": 0},
        },
        student_prompt_template=template,
        reasoning_effort="medium",
    )

    assert "generating assistant believed" not in record["student_prompt"]
    assert "generating assistant believed" in record["teacher_prompt"]
    assert "<established_facts>" in record["teacher_prompt"]
    assert record["index"] == "harm-pressure-knowledge-report:1"
