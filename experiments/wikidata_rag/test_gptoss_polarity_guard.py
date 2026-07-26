from experiments.wikidata_rag.label_gptoss_polarity_guard import (
    decisive_candidates,
    parse_json_object,
    prompt_for,
    validate_annotation,
)
from experiments.wikidata_rag.analyze_gptoss_polarity_guard import attach_labels


def record() -> dict:
    return {
        "question": "Who wrote The Moonstone?",
        "answer_full": "The Moonstone was written by Arthur Conan Doyle.",
        "candidate": {
            "id": "C00",
            "subject": "The Moonstone",
            "predicate": "author",
            "fact": "author: Wilkie Collins",
        },
    }


def test_derives_contradiction_from_grounded_comparison() -> None:
    parsed = {
        "id": "C00",
        "claim_quote": "The Moonstone was written by Arthur Conan Doyle.",
        "claimed_value": "Arthur Conan Doyle",
        "database_value": "Wilkie Collins",
        "entity_relation_match": True,
        "comparison": "mutually_exclusive",
        "reason": "Different authors answer the same slot.",
    }

    annotation, errors = validate_annotation(parsed, record())

    assert errors == []
    assert annotation is not None
    assert annotation["polarity"] == "contradicts"


def test_abstains_on_bad_source_span() -> None:
    parsed = {
        "id": "C00",
        "claim_quote": "Wilkie Collins",
        "claimed_value": "Wilkie Collins",
        "database_value": "Wilkie Collins",
        "entity_relation_match": True,
        "comparison": "same_value",
        "reason": "Copied the database instead of the response.",
    }

    annotation, errors = validate_annotation(parsed, record())

    assert "claim_quote" in errors
    assert annotation is not None and annotation["polarity"] is None


def test_abstains_when_relation_does_not_match() -> None:
    parsed = {
        "id": "C00",
        "claim_quote": "The Moonstone was written by Arthur Conan Doyle.",
        "claimed_value": "Arthur Conan Doyle",
        "database_value": "Wilkie Collins",
        "entity_relation_match": False,
        "comparison": "mutually_exclusive",
        "reason": "Wrong entity sense.",
    }

    annotation, errors = validate_annotation(parsed, record())

    assert errors == []
    assert annotation is not None and annotation["polarity"] is None


def test_vetoes_contradiction_for_multivalued_relation() -> None:
    value = record()
    value["candidate"] = value["candidate"] | {
        "predicate": "award received",
        "fact": "award received: Hugo Award",
    }
    value["answer_full"] = "The author received the Nebula Award."
    parsed = {
        "id": "C00",
        "claim_quote": "The author received the Nebula Award.",
        "claimed_value": "Nebula Award",
        "database_value": "Hugo Award",
        "entity_relation_match": True,
        "comparison": "mutually_exclusive",
        "reason": "Different awards.",
    }

    annotation, errors = validate_annotation(parsed, value)

    assert errors == []
    assert annotation is not None
    assert annotation["polarity"] is None
    assert annotation["polarity_veto"] == "nonexclusive_predicate"


def test_prompt_keeps_sources_separate_and_hides_polarity_label() -> None:
    prompt = prompt_for(record())

    assert "RESPONSE:" in prompt and "DATABASE EVIDENCE:" in prompt
    assert "Do not output a polarity label" in prompt
    assert "deceptive" not in prompt.lower()


def test_flattens_only_frozen_decisive_candidates() -> None:
    source = record() | {
        "question_group": "g",
        "deceptive": 1,
        "candidates": [
            record()["candidate"],
            record()["candidate"] | {"id": "C01"},
        ],
        "labels": [
            {"id": "C00", "label": "decisive"},
            {"id": "C01", "label": "irrelevant"},
        ],
    }

    flattened = decisive_candidates([source])

    assert len(flattened) == 1
    assert flattened[0]["candidate"]["id"] == "C00"


def test_parses_harmony_final_object() -> None:
    completion = 'assistantfinal\n{"id":"C00","comparison":"same_value"}'

    assert parse_json_object(completion) == {"id": "C00", "comparison": "same_value"}


def test_diagnostic_label_join_uses_exact_visible_content() -> None:
    rows = [record()]
    labeled = [record() | {"label": 1}]

    joined = attach_labels(rows, labeled)

    assert joined[0]["deceptive"] == 1
