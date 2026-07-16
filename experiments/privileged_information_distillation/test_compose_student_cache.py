from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.compose_student_cache import (
    compose_student_records,
)


def test_compose_student_records_preserves_target_and_replaces_prompt() -> None:
    prompt = {"dataset": "dataset", "index": 2, "label": 1, "student_prompt": "trace"}
    target = {
        "dataset": "dataset",
        "index": 2,
        "label": 1,
        "student_prompt": "ordinary",
        "student_target": "Prediction:1",
        "label_match": True,
    }

    result = compose_student_records(
        [prompt],
        [target],
        prompt_source="trace-cache",
        target_source="ordinary-cache",
    )

    assert result[0]["student_prompt"] == "trace"
    assert result[0]["student_target"] == "Prediction:1"
    assert result[0]["student_prompt_source"] == "trace-cache"
    assert result[0]["student_target_source"] == "ordinary-cache"


def test_compose_student_records_rejects_label_mismatch() -> None:
    prompt = {"dataset": "dataset", "index": 2, "label": 0, "student_prompt": "trace"}
    target = {"dataset": "dataset", "index": 2, "label": 1}

    with pytest.raises(ValueError, match="label mismatch"):
        compose_student_records(
            [prompt],
            [target],
            prompt_source="trace-cache",
            target_source="ordinary-cache",
        )
