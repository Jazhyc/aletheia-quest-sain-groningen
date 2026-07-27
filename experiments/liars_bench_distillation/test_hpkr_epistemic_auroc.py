import pytest

from experiments.liars_bench_distillation.evaluate_hpkr_epistemic_auroc import (
    evaluated_rows,
    generated_binary_scores,
    select_margin_arm,
)


class Output:
    def __init__(self, text: str):
        self.outputs = [type("Completion", (), {"text": text})()]


def condition(auroc: float) -> dict:
    return {"metrics": {"auroc": auroc}}


def test_generated_binary_scores_preserve_parse_contract() -> None:
    scores, errors = generated_binary_scores([
        Output("<reasoning_summary>x</reasoning_summary>\nPrediction:1"),
        Output("<reasoning_summary>y</reasoning_summary>\nPrediction:0"),
        Output("unfinished"),
    ])

    assert scores == [0.500001, 0.499999, 0.499999]
    assert errors == 1


def test_margin_selection_requires_material_auroc_gain() -> None:
    summaries = {
        "phoenix_control": condition(0.45),
        "epistemic_direct": condition(0.49),
        "epistemic_empty": condition(0.55),
        "epistemic_reasoning": condition(0.53),
    }

    selected = select_margin_arm(summaries, minimum_auroc_gain=0.05)

    assert selected["selected"] == "epistemic_empty"
    assert not selected["comparisons"]["epistemic_direct"]["passes"]
    assert selected["comparisons"]["epistemic_empty"]["passes"]


def test_evaluated_rows_keep_continuous_scores() -> None:
    records = [{
        "dataset": "hpkr",
        "index": "a",
        "category": "harm-pressure-knowledge-report",
        "source_model": "qwen",
        "label": 1,
    }]

    rows = evaluated_rows(records, [0.734])

    assert rows[0]["score"] == pytest.approx(0.734)
    assert rows[0]["label"] == 1
