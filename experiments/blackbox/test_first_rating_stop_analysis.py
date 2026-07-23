from experiments.blackbox.analyze_first_rating_stop import first_rating_stop_gate


def result(ba: float, instructed: float, varied: float, seconds: float, parses: int) -> dict:
    return {
        "metrics": {"balanced_accuracy": ba},
        "parse_errors": parses,
        "timing": {"score_seconds": seconds},
        "datasets": [
            {
                "dataset": "instructed-deception-a",
                "metrics": {"balanced_accuracy": instructed},
            },
            {
                "dataset": "varied-deception-b",
                "metrics": {"balanced_accuracy": varied},
            },
        ],
    }


def test_first_rating_stop_gate_checks_accuracy_parse_and_runtime() -> None:
    baseline = result(0.93, 0.98, 0.87, 100.0, 5)
    passing = result(0.93, 0.98, 0.87, 80.0, 5)
    slow = result(0.93, 0.98, 0.87, 90.0, 5)

    assert first_rating_stop_gate(baseline, passing)["gate"]["passed"] is True
    assert first_rating_stop_gate(baseline, slow)["gate"]["passed"] is False
