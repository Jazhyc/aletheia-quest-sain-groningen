import pytest
import numpy as np

from experiments.liars_bench_distillation.analyze_heavy_spectrum import compare
from experiments.liars_bench_distillation.evaluate_heavy_spectrum import (
    source_family,
    write_generation_checkpoint,
)


def result(macro: float, category: dict[str, float], cells: dict[str, float]) -> dict:
    return {
        "n": 800,
        "macro_category_ba": macro,
        "by_category": {key: {"balanced_accuracy": value} for key, value in category.items()},
        "by_category_family": {
            key: {"balanced_accuracy": value} for key, value in cells.items()
        },
    }


def test_spectrum_gate_checks_macro_category_and_cells() -> None:
    baseline = result(0.8, {"a": 0.8, "b": 0.8}, {"a/qwen": 0.8, "b/qwen": 0.8})
    passing = result(0.81, {"a": 0.82, "b": 0.80}, {"a/qwen": 0.83, "b/qwen": 0.79})
    failing = result(0.81, {"a": 0.84, "b": 0.78}, {"a/qwen": 0.86, "b/qwen": 0.74})

    assert compare(baseline, passing)["gate"]["passed"] is True
    assert compare(baseline, failing)["gate"]["passed"] is False
    assert compare(
        baseline,
        passing,
        minimum_macro_category_delta=0.02,
        minimum_category_delta=-0.03,
        minimum_category_family_delta=-0.05,
    )["gate"]["passed"] is False


@pytest.mark.parametrize(
    ("source_model", "family"),
    [
        ("gemma-3-27b-it", "gemma"),
        ("kimi-k2.6", "kimi"),
        ("llama-v3.3-70b-instruct", "llama"),
        ("mistral-small-3.1-24b-instruct", "mistral"),
        ("qwen-2.5-72b-instruct", "qwen"),
    ],
)
def test_spectrum_source_family_covers_full_artifact(source_model: str, family: str) -> None:
    assert source_family(source_model) == family


def test_generation_checkpoint_precedes_metric_aggregation(tmp_path) -> None:
    rows = [{
        "dataset": "d",
        "index": 7,
        "label": 1,
        "category": "c",
        "source_model": "kimi-k2.6",
    }]
    output = tmp_path / "generations.jsonl"

    write_generation_checkpoint(
        output,
        rows,
        [("details", "prompt")],
        np.asarray([[0.75]]),
        [{"generation": "Rating: 6"}],
    )

    parsed = __import__("json").loads(output.read_text())
    assert parsed["ensemble_member"] == "details"
    assert parsed["score"] == 0.75
    assert parsed["generation"] == "Rating: 6"
