from experiments.liars_bench_distillation.analyze_heavy_model_swap import (
    model_swap_gate,
)


def record(overall: float, families: dict[str, float]):
    return {
        "baseline": {"balanced_accuracy": overall},
        "baseline_by_family": {
            family: {"balanced_accuracy": value}
            for family, value in families.items()
        },
    }


def test_model_swap_gate_requires_overall_and_each_family_preservation() -> None:
    baseline = record(0.94, {"gemma": 0.94, "qwen": 0.95})
    assert model_swap_gate(
        baseline, record(0.935, {"gemma": 0.92, "qwen": 0.96})
    )["passed"]
    assert not model_swap_gate(
        baseline, record(0.95, {"gemma": 0.90, "qwen": 0.99})
    )["passed"]
