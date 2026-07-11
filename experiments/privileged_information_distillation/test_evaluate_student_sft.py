from experiments.privileged_information_distillation.evaluate_student_sft import parse_prediction


def test_parse_prediction_uses_last_explicit_prediction() -> None:
    text = "<reasoning_summary>Evidence.</reasoning_summary>\nPrediction:0\nPrediction: 1"
    assert parse_prediction(text) == 1


def test_parse_prediction_rejects_unstructured_binary_text() -> None:
    assert parse_prediction("The answer is 1.") is None
