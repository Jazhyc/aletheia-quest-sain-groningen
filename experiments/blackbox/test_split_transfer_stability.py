import pytest

from experiments.blackbox.analyze_split_transfer_stability import compare_transfer


def result(values: dict[str, float]) -> dict:
    return {
        "datasets": [
            {"dataset": key, "metrics": {"balanced_accuracy": value}}
            for key, value in values.items()
        ]
    }


def test_transfer_audit_counts_reversals_and_shared_wins() -> None:
    baseline = result({"a": 0.5, "b": 0.5, "c": 0.5})
    candidate_validation = result({"a": 0.6, "b": 0.4, "c": 0.5})
    candidate_test = result({"a": 0.7, "b": 0.6, "c": 0.5})

    audited = compare_transfer(
        baseline,
        candidate_validation,
        baseline,
        candidate_test,
        bootstrap_samples=100,
        seed=1,
    )

    assert audited["wins_both"] == 1
    assert audited["losses_both"] == 0
    assert audited["direction_reversals"] == 1
    assert audited["validation_zero_units"] == 1
    assert audited["exact_sign_agreement_fraction"] == pytest.approx(2 / 3)


def test_transfer_audit_rejects_mismatched_units() -> None:
    with pytest.raises(ValueError, match="keys differ"):
        compare_transfer(
            result({"a": 0.5}),
            result({"b": 0.6}),
            result({"a": 0.5}),
            result({"a": 0.6}),
            bootstrap_samples=10,
        )
