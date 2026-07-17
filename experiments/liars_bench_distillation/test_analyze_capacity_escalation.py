from experiments.liars_bench_distillation.analyze_capacity_escalation import analyze


def rows(member_votes: dict[str, list[int]], *, candidate: bool) -> list[dict]:
    labels = [0, 0, 1, 1]
    output = []
    for member, votes in member_votes.items():
        for index, (label, vote) in enumerate(zip(labels, votes, strict=True)):
            output.append({
                "dataset": "unit",
                "index": index,
                "label": label,
                "category": "unit",
                "source_model": "qwen-test",
                "ensemble_member": member,
                "rating": 2 if vote else 1,
                "parse_error": False,
                "candidate": candidate,
            })
    return output


def test_escalation_only_uses_baseline_disagreement() -> None:
    baseline = rows({
        "a": [0, 1, 0, 1],
        "b": [0, 0, 0, 1],
        "c": [0, 0, 0, 1],
    }, candidate=False)
    candidate = rows({
        "a": [1, 0, 1, 0],
        "b": [1, 0, 1, 0],
        "c": [1, 0, 1, 0],
    }, candidate=True)

    result = analyze(
        baseline,
        candidate,
        macro_field="category",
        minimum_macro_delta=-1.0,
        minimum_group_delta=-1.0,
        minimum_category_family_delta=-1.0,
        maximum_query_fraction=1.0,
    )

    assert result["q27_query_rows"] == 1
    assert result["q27_query_fraction"] == 0.25
    assert result["changed_rows"] == 1
    assert result["conditions"]["baseline"]["macro"]["balanced_accuracy"] == 0.5
    assert result["conditions"]["cascade"]["macro"]["balanced_accuracy"] == 0.75
    assert result["gate"]["passed"] is True
