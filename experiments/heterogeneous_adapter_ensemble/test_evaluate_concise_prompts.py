from experiments.heterogeneous_adapter_ensemble.evaluate_concise_prompts import (
    CONDITIONS,
    prompt_for_condition,
)


BASE = "Judge carefully.\n\nOutput exactly:\nPrediction:<0 or 1>"


def test_all_conditions_preserve_prediction_after_reasoning_contract() -> None:
    for condition in CONDITIONS:
        prompt = prompt_for_condition(BASE, "deception", condition)
        assert prompt.count("Output exactly:") == 1
        assert prompt.endswith("Prediction:<0 or 1>")
        if condition != "baseline":
            assert prompt.index("reasoning") < prompt.index("Output exactly:")


def test_member_specific_contrast_prompt() -> None:
    deception = prompt_for_condition(BASE, "deception", "contrast24")
    intent = prompt_for_condition(BASE, "resolved_intent", "contrast24")
    assert "claim and its correction" in deception
    assert "final operative decision" in intent
