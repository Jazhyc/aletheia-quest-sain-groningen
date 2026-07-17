from experiments.blackbox.analyze_qwen27b_selfconsistency import (
    BASELINE_INSTRUCTED_BA,
    BASELINE_OVERALL_BA,
    BASELINE_VARIED_BA,
    self_consistency_gate,
)


def test_self_consistency_gate_requires_accuracy_diversity_and_parse_stability() -> None:
    passing = self_consistency_gate(
        BASELINE_OVERALL_BA + 0.004,
        BASELINE_INSTRUCTED_BA,
        BASELINE_VARIED_BA,
        5,
        0.05,
    )
    assert passing["passed"]
    assert passing["test_authorized"]
    assert not self_consistency_gate(
        BASELINE_OVERALL_BA,
        BASELINE_INSTRUCTED_BA,
        BASELINE_VARIED_BA,
        6,
        0.05,
    )["passed"]
    assert not self_consistency_gate(
        BASELINE_OVERALL_BA,
        BASELINE_INSTRUCTED_BA,
        BASELINE_VARIED_BA,
        0,
        0.049,
    )["passed"]
