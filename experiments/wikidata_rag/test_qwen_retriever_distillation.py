from experiments.wikidata_rag.build_qwen_retriever_distillation import (
    build_pairs,
    group_bucket,
    retrieval_prompt,
)


def teacher_row() -> dict:
    return {
        "dataset": "dataset",
        "index": 7,
        "question_group": "group",
        "question": "Who wrote the book?",
        "answer_full": "Arthur wrote the book.",
        "parse_error": False,
        "candidates": [
            {
                "id": "C00",
                "subject": "The book",
                "fact": "author: Wilkie",
                "popularity": 10,
            },
            {
                "id": "C01",
                "subject": "The book",
                "fact": "publication date: 1868",
                "popularity": 20,
            },
            {
                "id": "C02",
                "subject": "Another book",
                "fact": "author: Arthur",
                "popularity": 30,
            },
        ],
        "labels": [
            {"id": "C00", "label": "decisive"},
            {"id": "C01", "label": "relevant_insufficient"},
            {"id": "C02", "label": "irrelevant"},
        ],
    }


def test_retrieval_prompt_is_label_blind_and_requires_direct_sufficiency() -> None:
    prompt = retrieval_prompt(teacher_row(), teacher_row()["candidates"][0])

    assert "BY ITSELF" in prompt
    assert "Agreement and contradiction are both 1" in prompt
    assert "author: Wilkie" in prompt
    assert "decisive" not in prompt.lower()


def test_build_pairs_prefers_planner_false_positive_as_hard_negative() -> None:
    row = teacher_row()
    planner = [{
        "dataset": "dataset",
        "index": 7,
        "selected": [{
            "id": "C02",
            "claim_quote": "Arthur wrote",
            "relation": "supports",
        }],
        "parse_error": False,
    }]

    records, report = build_pairs(
        [row],
        planner,
        fit_buckets={group_bucket("group")},
    )

    assert [record["label"] for record in records] == [1, 0]
    assert records[0]["candidate_id"] == "C00"
    assert records[1]["candidate_id"] == "C02"
    assert records[1]["hard_negative_from_planner"] is True
    assert report["counts"]["pairs"] == 1
    assert report["counts"]["planner_false_positive_negatives"] == 1


def test_build_pairs_falls_back_to_relevant_insufficient_negative() -> None:
    records, _ = build_pairs(
        [teacher_row()],
        [],
        fit_buckets={group_bucket("group")},
    )

    assert records[1]["candidate_id"] == "C01"
    assert records[1]["candidate_teacher_label"] == "relevant_insufficient"


def test_build_pairs_respects_frozen_question_group_buckets() -> None:
    bucket = group_bucket("group")
    other_bucket = next(value for value in range(5) if value != bucket)

    records, report = build_pairs(
        [teacher_row()],
        [],
        fit_buckets={other_bucket},
    )

    assert records == []
    assert report["counts"][f"teacher_rows_bucket_{bucket}"] == 1


def test_build_pairs_retains_planner_false_positive_on_negative_only_row() -> None:
    source = teacher_row()
    source["labels"][0]["label"] = "relevant_insufficient"
    planner = [{
        "dataset": "dataset",
        "index": 7,
        "selected": [{
            "id": "C00",
            "claim_quote": "Arthur wrote",
            "relation": "contradicts",
        }],
        "parse_error": False,
    }]

    records, report = build_pairs(
        [source],
        planner,
        fit_buckets={group_bucket("group")},
    )

    assert len(records) == 1
    assert records[0]["candidate_id"] == "C00"
    assert records[0]["label"] == 0
    assert report["counts"]["planner_false_positive_negative_anchors"] == 1
