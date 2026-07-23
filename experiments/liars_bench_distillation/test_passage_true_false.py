from experiments.liars_bench_distillation.prompts import PASSAGE_TRUE_FALSE_PROMPT
from experiments.liars_bench_distillation.router import is_passage_true_false_request


def test_passage_specialist_contract_is_source_grounded() -> None:
    assert "authoritative source" in PASSAGE_TRUE_FALSE_PROMPT
    assert "outside knowledge" in PASSAGE_TRUE_FALSE_PROMPT
    assert "Prediction:<0 or 1>" in PASSAGE_TRUE_FALSE_PROMPT


def test_passage_router_accepts_boolean_punctuation() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "=== BEGIN PASSAGE ===\nA is B.\n=== END PASSAGE ===\n"
                "True or False? Is A B?"
            ),
        },
        {"role": "assistant", "content": "True."},
    ]
    assert is_passage_true_false_request(messages)
