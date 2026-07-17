import pytest

from experiments.liars_bench_distillation.analyze_capacity_bootstrap import (
    analyze,
    parse_partition,
)


def rows(predictions):
    result = []
    for index, (label, prediction) in enumerate(predictions):
        result.append({
            "dataset": "d",
            "index": index,
            "category": "c",
            "label": label,
            "prediction": prediction,
        })
    return result


def test_capacity_bootstrap_counts_paired_fixes_and_breaks() -> None:
    baseline = rows([(0, 1), (0, 0), (1, 0), (1, 1)])
    candidate = rows([(0, 0), (0, 1), (1, 1), (1, 0)])
    result = analyze(baseline, candidate, samples=100, seed=1)
    assert result["changed"] == 4
    assert result["fixes"] == 2
    assert result["breaks"] == 2


def test_capacity_bootstrap_rejects_misaligned_rows() -> None:
    baseline = rows([(0, 0), (1, 1)])
    candidate = rows([(0, 0), (1, 1)])
    candidate[1]["index"] = 9
    with pytest.raises(ValueError, match="identities"):
        analyze(baseline, candidate, samples=10)


def test_parse_partition_separates_clean_decision_gains() -> None:
    baseline = rows(
        [(0, 1), (0, 0), (1, 0), (1, 1)]
        + [(0, 1), (0, 0), (1, 0), (1, 1)]
    )
    candidate = rows(
        [(0, 0), (0, 0), (1, 1), (1, 1)]
        + [(0, 0), (0, 0), (1, 1), (1, 1)]
    )
    baseline_generations = [
        {
            "dataset": "d",
            "index": index,
            "parse_error": member == 0 and index < 4,
        }
        for member in range(3)
        for index in range(8)
    ]
    candidate_generations = [
        {"dataset": "d", "index": index, "parse_error": False}
        for _member in range(3)
        for index in range(8)
    ]
    result = parse_partition(
        baseline, candidate, baseline_generations, candidate_generations
    )
    assert result["baseline_rows_with_any_parse_error"] == 4
    assert result["candidate_rows_with_any_parse_error"] == 0
    assert result["partitions"]["baseline_clean"]["fixes"] == 2
    assert result["partitions"]["baseline_any_parse_error"]["fixes"] == 2
