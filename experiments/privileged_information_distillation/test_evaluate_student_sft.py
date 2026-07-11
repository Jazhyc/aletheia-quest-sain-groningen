import json

from experiments.privileged_information_distillation.evaluate_student_sft import (
    load_retrieval_cache,
    parse_prediction,
)


def test_parse_prediction_uses_last_explicit_prediction() -> None:
    text = "<reasoning_summary>Evidence.</reasoning_summary>\nPrediction:0\nPrediction: 1"
    assert parse_prediction(text) == 1


def test_parse_prediction_rejects_unstructured_binary_text() -> None:
    assert parse_prediction("The answer is 1.") is None


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
