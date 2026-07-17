from experiments.liars_bench_distillation.analyze_adapter_heavy_preservation import (
    compare_adapter_heavy,
)


def result(macro: float, category: float, cell: float, parses: int) -> dict:
    return {
        "n": 800,
        "macro_category_ba": macro,
        "by_category": {"a": {"balanced_accuracy": category}},
        "by_category_family": {"a/qwen": {"balanced_accuracy": cell}},
        "parse_errors": parses,
    }


def test_adapter_heavy_gate_checks_accuracy_cells_and_parse() -> None:
    baseline = result(0.80, 0.80, 0.80, 160)
    passing = result(0.795, 0.79, 0.76, 181)
    bad_cell = result(0.80, 0.80, 0.74, 160)

    assert compare_adapter_heavy(baseline, passing)["gate"]["passed"] is True
    assert compare_adapter_heavy(baseline, bad_cell)["gate"]["passed"] is False
