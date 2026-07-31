import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kimi_liars_enrichment.export_hf_dataset import (
    ASSET_DIR,
    EXPECTED_ROWS,
    write_dataset,
)


HUB_CACHE = Path(
    os.environ.get(
        "HF_HUB_CACHE",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        / "hub",
    )
)
RAW_ARTIFACT = (
    ROOT
    / "results/blackbox/kimi_k3_tvg_soft_full_plus_liars_v1/train/student_rows.jsonl"
)
HAS_FROZEN_LOCAL_ARTIFACTS = RAW_ARTIFACT.is_file() and HUB_CACHE.is_dir()


def load_renderer():
    path = ASSET_DIR / "render_prompts.py"
    spec = importlib.util.spec_from_file_location("phoenix81_hf_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not HAS_FROZEN_LOCAL_ARTIFACTS,
    reason="frozen ignored Phoenix 8.1 caches are not present",
)
def test_export_is_mit_annotation_only(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path, HUB_CACHE)
    frame = pd.read_parquet(tmp_path / "data/train-00000-of-00001.parquet")
    assert len(frame) == EXPECTED_ROWS
    assert manifest["rows"] == EXPECTED_ROWS
    assert manifest["row_counts"] == {
        "aletheias_quest": 6573,
        "liars_bench": 6576,
    }
    assert not {
        "messages",
        "label",
        "deceptive",
        "student_prompt",
        "teacher_prompt",
    } & set(frame.columns)
    assert frame["row_id"].nunique() == EXPECTED_ROWS
    assert (frame["soft_target_deceptive"] == frame["teacher_probability_1"]).all()
    assert (
        (frame["teacher_probability_0"] + frame["teacher_probability_1"] - 1.0)
        .abs()
        .max()
        < 1e-12
    )


@pytest.mark.skipif(
    not HAS_FROZEN_LOCAL_ARTIFACTS,
    reason="frozen ignored Phoenix 8.1 caches are not present",
)
def test_standalone_renderer_matches_frozen_liars_prompts() -> None:
    import json

    renderer = load_renderer()
    templates = json.loads((ASSET_DIR / "prompt_templates.json").read_text()) if (
        ASSET_DIR / "prompt_templates.json"
    ).exists() else None
    # The templates file is generated into the upload directory, so use the
    # canonical repository constants for this source-level renderer test.
    if templates is None:
        from experiments.kimi_liars_enrichment.prompts import (
            HARM_PRESSURE_CHOICE_BINARY_PROMPT,
        )
        from experiments.openrouter_qwen397_tvg.run_openrouter_tvg import (
            FROZEN_PROMPT_TEMPLATE,
        )
        from submission.phoenix_wright import (
            ACTION_PROMPT,
            DECEPTION_PROMPT,
            EPISTEMIC_PROMPT,
        )

        templates = {
            "ordinary": DECEPTION_PROMPT,
            "ordinary_kimi_query": FROZEN_PROMPT_TEMPLATE,
            "epistemic": EPISTEMIC_PROMPT,
            "action": ACTION_PROMPT,
            "harm_pressure_choice": HARM_PRESSURE_CHOICE_BINARY_PROMPT,
        }
    students_path = (
        ROOT
        / "results/blackbox/kimi_k3_liars_semantic_soft_v1/train/student_rows.jsonl"
    )
    generations_path = students_path.with_name("generations.jsonl")
    students = [json.loads(line) for line in students_path.open()]
    generations = {
        (str(row["dataset"]), str(row["index"])): row
        for row in (json.loads(line) for line in generations_path.open())
    }
    for student in students[::257]:
        category = str(student["category"])
        student_kind = {
            "harm-pressure-choice": "ordinary",
            "harm-pressure-knowledge-report": "epistemic",
            "insider-trading": "action",
            "soft-trigger": "ordinary",
        }[category]
        student_chars = {
            "harm-pressure-choice": 3000,
            "harm-pressure-knowledge-report": 6000,
            "insider-trading": 4500,
            "soft-trigger": 3000,
        }[category]
        student_truncation = (
            "head_tail"
            if category in {"harm-pressure-knowledge-report", "insider-trading"}
            else "tail"
        )
        rendered_student = renderer.render_prompt(
            student["messages"],
            templates[student_kind],
            student_chars,
            student_truncation,
        )
        assert rendered_student == student["student_prompt"]

        condition = str(student["teacher_prompt_kind"])
        teacher_kind = (
            "harm_pressure_choice"
            if (category, condition) == ("harm-pressure-choice", "semantic")
            else "epistemic"
            if (category, condition)
            == ("harm-pressure-knowledge-report", "semantic")
            else "ordinary"
        )
        teacher_chars = 6000 if teacher_kind == "epistemic" else 3000
        teacher_truncation = "head_tail" if teacher_kind == "epistemic" else "tail"
        rendered_teacher = renderer.render_prompt(
            student["messages"],
            templates[teacher_kind],
            teacher_chars,
            teacher_truncation,
        )
        generation = generations[(str(student["dataset"]), str(student["index"]))]
        assert hashlib.sha256(rendered_teacher.encode()).hexdigest() == generation[
            "prompt_sha256"
        ]
