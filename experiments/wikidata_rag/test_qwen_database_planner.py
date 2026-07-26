from experiments.wikidata_rag.qwen_database_planner import (
    evaluate_plans,
    grounded_quote,
    prompt_for,
    validate_plan,
)
from experiments.wikidata_rag.build_qwen_planner_sweep_cache import build_cache


def candidate_row() -> dict:
    return {
        "dataset": "dataset",
        "index": 1,
        "question": "Who wrote the book?",
        "answer_full": "The book was written by Arthur. It appeared in 1868.",
        "question_group": "group",
        "currently_covered": False,
        "candidates": [
            {"id": "C00", "subject": "The book", "fact": "author: Wilkie"},
            {"id": "C01", "subject": "The book", "fact": "publication date: 1868"},
        ],
    }


def test_prompt_demands_bounded_database_evidence() -> None:
    prompt = prompt_for(candidate_row())

    assert "You may only select candidate IDs" in prompt
    assert "Check material supporting details" in prompt
    assert "C00 | The book | author: Wilkie" in prompt


def test_validate_plan_accepts_grounded_bounded_selection() -> None:
    parsed = [{
        "id": "C00",
        "claim_quote": "written by Arthur",
        "relation": "contradicts",
    }]

    selected, errors = validate_plan(parsed, candidate_row())

    assert errors == []
    assert selected == parsed


def test_validate_plan_rejects_unknown_and_ungrounded_selection() -> None:
    parsed = [
        {"id": "C99", "claim_quote": "written by Arthur", "relation": "contradicts"},
        {"id": "C01", "claim_quote": "published in 1900", "relation": "supports"},
    ]

    selected, errors = validate_plan(parsed, candidate_row())

    assert selected == []
    assert "unknown:C99" in errors
    assert "ungrounded_quote:C01" in errors


def test_validate_plan_rejects_extra_fields_and_excess_selections() -> None:
    row = candidate_row()
    extra = [{
        "id": "C00",
        "claim_quote": "written by Arthur",
        "relation": "contradicts",
        "reason": "not allowed",
    }]
    selected, errors = validate_plan(extra, row)
    assert selected == []
    assert errors == ["fields"]

    selected, errors = validate_plan([extra[0]] * 4, row)
    assert selected == []
    assert errors == ["too_many"]


def test_grounded_quote_requires_nontrivial_exact_content() -> None:
    assert grounded_quote("written by Arthur", candidate_row()["answer_full"])
    assert not grounded_quote("by", candidate_row()["answer_full"])
    assert not grounded_quote("written by Wilkie", candidate_row()["answer_full"])


def test_evaluate_plans_reports_precision_recall_and_novelty() -> None:
    plan = candidate_row() | {
        "selected": [{
            "id": "C00",
            "claim_quote": "written by Arthur",
            "relation": "contradicts",
        }],
        "parse_error": False,
    }
    teacher = candidate_row() | {
        "parse_error": False,
        "labels": [
            {"id": "C00", "label": "decisive"},
            {"id": "C01", "label": "relevant_insufficient"},
        ],
    }

    report = evaluate_plans(
        [plan],
        [teacher],
        training_question_groups={"different"},
    )

    assert report["all"]["selected_fact_precision"] == 1.0
    assert report["all"]["decisive_row_recall"] == 1.0
    assert report["all"]["new_retrievals_outside_rule_coverage"] == 1
    assert report["novel_questions"]["rows"] == 1


def test_build_cache_uses_valid_selection_and_cross_dataset_noise() -> None:
    first = candidate_row() | {
        "dataset": "a",
        "selected": [{
            "id": "C00",
            "claim_quote": "written by Arthur",
            "relation": "contradicts",
        }],
        "parse_error": False,
    }
    second = candidate_row() | {
        "dataset": "b",
        "index": 2,
        "selected": [{
            "id": "C01",
            "claim_quote": "appeared in 1868",
            "relation": "supports",
        }],
        "parse_error": False,
    }
    inactive = candidate_row() | {
        "dataset": "c",
        "index": 3,
        "selected": [],
        "parse_error": False,
    }

    cache = build_cache([first, second, inactive])

    assert cache[0]["selected_candidate"]["planner_relation"] == "contradicts"
    assert cache[0]["real_passages"][0]["text"] == "author: Wilkie"
    assert cache[0]["shuffled_from"] == {"dataset": "b", "index": 2}
    assert cache[1]["shuffled_from"] == {"dataset": "a", "index": 1}
    assert cache[2]["real_passages"] == []
    assert cache[2]["shuffled_passages"] == []


def test_build_cache_preserves_multiple_selected_facts() -> None:
    first = candidate_row() | {
        "dataset": "a",
        "selected": [
            {
                "id": "C00",
                "claim_quote": "written by Arthur",
                "relation": "contradicts",
            },
            {
                "id": "C01",
                "claim_quote": "appeared in 1868",
                "relation": "supports",
            },
        ],
        "parse_error": False,
    }
    donor = candidate_row() | {
        "dataset": "b",
        "index": 2,
        "selected": [{
            "id": "C00",
            "claim_quote": "written by Arthur",
            "relation": "contradicts",
        }],
        "parse_error": False,
    }

    cache = build_cache([first, donor])

    evidence = cache[0]["real_passages"][0]
    assert evidence["title"] == "Selected Wikidata facts"
    assert "The book: author: Wilkie" in evidence["text"]
    assert "The book: publication date: 1868" in evidence["text"]
    assert len(cache[0]["selected_candidate"]["selections"]) == 2
