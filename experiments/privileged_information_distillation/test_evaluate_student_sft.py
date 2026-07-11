import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.evaluate_student_sft import (
    COUNTERFACTUAL_STRICT_RE,
    PREDICTION_ONLY_STRICT_RE,
    load_retrieval_cache,
    parse_prediction,
)


def test_parse_prediction_uses_last_explicit_prediction() -> None:
    text = "<reasoning_summary>Evidence.</reasoning_summary>\nPrediction:0\nPrediction: 1"
    assert parse_prediction(text) == 1


def test_parse_prediction_rejects_unstructured_binary_text() -> None:
    assert parse_prediction("The answer is 1.") is None


def test_parse_prediction_accepts_literal_placeholder_brackets() -> None:
    assert parse_prediction("Prediction:<0>") == 0


def test_load_retrieval_cache_formats_passages(tmp_path) -> None:
    path = tmp_path / "rag.jsonl"
    path.write_text(json.dumps({
        "dataset": "dataset",
        "index": 7,
        "passages": [{"title": "Cerberus", "text": "Three-headed guardian."}],
    }) + "\n")

    assert load_retrieval_cache(path) == {
        ("dataset", 7): "- Cerberus: Three-headed guardian."
    }


def test_counterfactual_strict_format_requires_ordered_fields() -> None:
    valid = (
        "<reasoning_summary>Summary.</reasoning_summary>\n"
        "<facts>Facts.</facts>\n"
        "<contradiction>No contradiction.</contradiction>\nPrediction:0"
    )

    assert COUNTERFACTUAL_STRICT_RE.fullmatch(valid)
    assert not COUNTERFACTUAL_STRICT_RE.fullmatch(
        valid.replace("<facts>Facts.</facts>\n", "")
    )


def test_prediction_only_strict_format_accepts_binary_contract() -> None:
    assert PREDICTION_ONLY_STRICT_RE.fullmatch("Prediction:1")
    assert PREDICTION_ONLY_STRICT_RE.fullmatch("Prediction:<0>")
    assert not PREDICTION_ONLY_STRICT_RE.fullmatch("Reasoning\nPrediction:1")
