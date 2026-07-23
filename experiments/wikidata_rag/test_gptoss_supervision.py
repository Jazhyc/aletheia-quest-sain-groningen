from experiments.wikidata_rag.label_gptoss_retrieval_candidates import (
    parse_json_array,
    prompt_for,
    validate_labels,
)
from experiments.wikidata_rag.analyze_gptoss_supervision import duplicate_question_consistency
from experiments.wikidata_rag.learn_gptoss_relation_rules import phrase_features, wilson_lower


def row() -> dict:
    return {
        "answer_full": "The answer is Paris, which is in France.",
        "candidates": [{"id": "C00"}, {"id": "C01"}],
    }


def test_parse_and_validate_grounded_labels() -> None:
    completion = '''assistantfinal
    [{"id":"C00","label":"supports","claim_quote":"Paris","reason":"direct"},
     {"id":"C01","label":"irrelevant","claim_quote":"NONE","reason":"wrong slot"}]
    '''
    labels, errors = validate_labels(parse_json_array(completion), row())

    assert errors == []
    assert [item["label"] for item in labels] == ["supports", "irrelevant"]


def test_rejects_ungrounded_and_missing_labels() -> None:
    completion = '''[{"id":"C00","label":"contradicts",
      "claim_quote":"Berlin","reason":"not grounded"}]'''
    labels, errors = validate_labels(parse_json_array(completion), row())

    assert labels[0]["id"] == "C00"
    assert "ungrounded_quote:C00" in errors
    assert "missing:C01" in errors


def test_decisiveness_mode_needs_no_claim_quote() -> None:
    completion = '''[{"id":"C00","label":"decisive","reason":"direct author"},
      {"id":"C01","label":"relevant_insufficient","reason":"wrong relation"}]'''
    labels, errors = validate_labels(parse_json_array(completion), row(), "decisive")

    assert errors == []
    assert [item["label"] for item in labels] == ["decisive", "relevant_insufficient"]
    assert all("claim_quote" not in item for item in labels)


def test_decisiveness_prompt_avoids_polarity_target() -> None:
    value = row() | {
        "question": "Who wrote the book?", "candidates": [
            {"id": "C00", "subject": "The book", "fact": "author: Wilkie"},
            {"id": "C01", "subject": "The book", "fact": "inception: 1868"},
        ],
    }
    prompt = prompt_for(value, "decisive")

    assert "both are decisive" in prompt
    assert "claim_quote" not in prompt


def test_duplicate_question_consistency_counts_decisive_conflicts() -> None:
    base = {
        "question_group": "same", "candidates": [
            {"id": "C00", "qid": "Q1", "fact": "author: A"}
        ],
    }
    report = duplicate_question_consistency([
        base | {"labels": [{"id": "C00", "label": "decisive"}]},
        base | {"labels": [{"id": "C00", "label": "relevant_insufficient"}]},
    ])

    assert report["overlapping_candidate_pairs"] == 1
    assert report["decisive_conflicts"] == 1
    assert report["decisive_binary_agreement"] == 0.0


def test_relation_rule_features_retain_function_words() -> None:
    features = phrase_features("The Moonstone was written by whom?")

    assert "g2=by|whom" in features
    assert 0.0 < wilson_lower(3, 3) < 1.0
