from experiments.blackbox.audit_qwen27_deployment_shape import summarize


def test_summarize_prompt_lengths_is_integer_and_empty_safe() -> None:
    assert summarize([]) == {"n": 0, "median": 0, "p95": 0, "max": 0}
    assert summarize([10, 20, 30]) == {
        "n": 3,
        "median": 20,
        "p95": 29,
        "max": 30,
    }
