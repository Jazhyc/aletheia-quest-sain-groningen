from experiments.liars_bench_distillation.analyze_ensemble_votes import aggregate_generations


def test_vote_rules_keep_parse_failures_negative() -> None:
    rows = []
    labels = [0, 0, 1, 1]
    member_ratings = {
        "a": [1, 2, 2, None],
        "b": [1, 1, 2, 2],
        "c": [1, 1, 1, 2],
    }
    for member, ratings in member_ratings.items():
        for index, (label, rating) in enumerate(zip(labels, ratings, strict=True)):
            rows.append({
                "dataset": "unit",
                "index": index,
                "label": label,
                "ensemble_member": member,
                "rating": rating,
                "parse_error": rating is None,
            })

    result = aggregate_generations(rows, stratum_field="dataset")

    assert result["n"] == 4
    assert result["parse_errors_by_member"] == {"a": 1, "b": 0, "c": 0}
    assert result["disagreement_rows"] == 3
    assert result["rules"]["or"]["macro"]["balanced_accuracy"] == 0.75
    assert result["rules"]["majority"]["macro"]["balanced_accuracy"] == 1.0
    assert result["rules"]["unanimous"]["macro"]["balanced_accuracy"] == 0.5
