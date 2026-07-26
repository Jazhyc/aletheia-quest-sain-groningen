from experiments.wikidata_rag.build_polarity_distillation import (
    build_pairs,
    polarity_prompt,
    replace_database_value,
)
from experiments.wikidata_rag.build_qwen_retriever_distillation import group_bucket


def contradiction_row() -> dict:
    group = next(
        name for name in (f"group-{index}" for index in range(100))
        if group_bucket(name) in {2, 3, 4}
    )
    return {
        "source_row_index": 7,
        "question_group": group,
        "question": "Who wrote The Moonstone?",
        "answer_full": "It was written by Arthur Conan Doyle.",
        "candidate": {
            "id": "C00",
            "subject": "The Moonstone",
            "predicate": "author",
            "fact": "author: Wilkie Collins",
        },
        "annotation": {
            "polarity": "contradicts",
            "claimed_value": "Arthur Conan Doyle",
            "database_value": "Wilkie Collins",
        },
    }


def test_replaces_grounded_value_in_fact() -> None:
    candidate = contradiction_row()["candidate"]

    twin = replace_database_value(candidate, "Wilkie Collins", "Arthur Conan Doyle")

    assert twin is not None
    assert twin["fact"] == "author: Arthur Conan Doyle"
    assert candidate["fact"] == "author: Wilkie Collins"


def test_builds_balanced_real_counterfactual_pair() -> None:
    records, report = build_pairs([contradiction_row()])

    assert report["pairs"] == 1
    assert [record["label"] for record in records] == [1, 0]
    assert records[0]["dataset"] == records[1]["dataset"]
    assert "Wilkie Collins" in records[0]["student_prompt"]
    assert "Arthur Conan Doyle" in records[1]["student_prompt"]


def test_prompt_does_not_expose_hidden_deception_label() -> None:
    row = contradiction_row()
    prompt = polarity_prompt(row, row["candidate"])

    assert "Predict 1" in prompt
    assert "deception label" not in prompt.lower()
