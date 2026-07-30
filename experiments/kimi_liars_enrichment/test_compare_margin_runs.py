from experiments.kimi_liars_enrichment.compare_margin_runs import compare


def test_compare_margin_runs_reports_exact_and_ordering_parity() -> None:
    reference = {
        ("a", "0"): 0.1,
        ("a", "1"): 0.5,
        ("b", "0"): 0.9,
    }
    exact = compare(reference, dict(reference))
    assert exact["exact_scores"] == 3
    assert exact["ordering_changes"] == 0
    assert (
        exact["reference_float32_digest"]
        == exact["candidate_float32_digest"]
    )

    changed = compare(reference, {**reference, ("a", "1"): 0.95})
    assert changed["exact_scores"] == 2
    assert changed["ordering_changes"] == 1
