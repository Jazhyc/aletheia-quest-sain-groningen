from __future__ import annotations

import sqlite3

from experiments.compact_wikipedia_index.build_validation_cache import (
    add_matched_shuffle,
    atomic_queries,
    raw_queries,
)
from experiments.compact_wikipedia_index.generate_claim_queries import parse_claims

from experiments.compact_wikipedia_index.core import (
    LinearReranker,
    adjacent_window,
    create_index,
    fts_expression,
    iter_page_sentences,
    reference_source,
    retrieve,
    retrieve_proposition,
    source_matches,
)


def test_iter_page_sentences_deduplicates_and_skips_errors() -> None:
    pages = [
        {
            "canonical_title": "Paris",
            "url": "https://example.test/paris",
            "extract": "Paris is the capital city of France. Paris has many museums.",
        },
        {
            "canonical_title": "Paris",
            "url": "https://example.test/paris",
            "extract": "Paris is the capital city of France.",
        },
        {"canonical_title": "Missing", "extract": "Long enough sentence.", "missing": True},
    ]
    assert [row["text"] for row in iter_page_sentences(pages, minimum_chars=10)] == [
        "Paris is the capital city of France.",
        "Paris has many museums.",
    ]


def test_fts_expression_uses_content_terms() -> None:
    assert fts_expression("What is the capital of France?") == '"what" OR "capital" OR "france"'


def test_retrieve_ranks_exact_entity_and_number() -> None:
    connection = sqlite3.connect(":memory:")
    create_index(
        connection,
        [
            {
                "title": "Summer Olympics",
                "url": "u1",
                "sentence_index": 0,
                "text": "The 2016 Summer Olympics were held in Rio de Janeiro.",
            },
            {
                "title": "Tokyo",
                "url": "u2",
                "sentence_index": 0,
                "text": "Tokyo hosted the 2020 Summer Olympics.",
            },
        ],
    )
    rows = retrieve(
        connection,
        "The 2016 Summer Olympics were held in Rio de Janeiro.",
        limit=2,
    )
    assert rows[0]["title"] == "Summer Olympics"
    assert rows[0]["number_recall"] == 1.0


def test_proposition_retrieval_unions_claim_and_question_views() -> None:
    connection = sqlite3.connect(":memory:")
    create_index(
        connection,
        [
            {
                "title": "Apollo 11",
                "url": "u1",
                "sentence_index": 0,
                "text": "Apollo 11 landed on the Moon in 1969.",
            },
            {
                "title": "Apollo program",
                "url": "u2",
                "sentence_index": 0,
                "text": "The Apollo program conducted several crewed missions.",
            },
        ],
    )
    rows = retrieve_proposition(
        connection,
        "When did Apollo 11 reach the Moon?",
        "Apollo 11 landed on the Moon in 1969.",
        limit=2,
    )
    assert rows[0]["title"] == "Apollo 11"
    assert len(rows[0]["features"]) == 11


def test_linear_reranker_and_adjacent_window() -> None:
    reranker = LinearReranker(
        mean=(0.0,) * 11,
        scale=(1.0,) * 11,
        coefficients=(1.0,) + (0.0,) * 10,
        intercept=0.0,
        threshold=0.5,
    )
    assert reranker.probability((1.0,) + (0.0,) * 10) > 0.5

    connection = sqlite3.connect(":memory:")
    create_index(
        connection,
        [
            {"title": "Paris", "url": "u", "sentence_index": 0, "text": "Paris is in France."},
            {"title": "Paris", "url": "u", "sentence_index": 1, "text": "It lies on the Seine."},
        ],
    )
    assert adjacent_window(
        connection,
        {"title": "Paris", "sentence_index": 0},
    ) == "Paris is in France. It lies on the Seine."


def test_source_matching_accepts_a_sentence_inside_window() -> None:
    candidate = {"title": "Paris", "text": "Paris is the capital of France."}
    reference = {
        "title": "Paris",
        "text": "Claim: Paris is in France.\nSource sentence: Paris is the capital of France. It lies on the Seine.",
    }
    assert reference_source(reference["text"]) == (
        "Paris is the capital of France. It lies on the Seine."
    )
    assert source_matches(candidate, reference)


def test_query_modes_keep_raw_mode_teacher_blind() -> None:
    record = {
        "output": "Paris is in France. It lies on the Seine.",
        "claims": [{
            "quote": "Paris is in France.",
            "proposition": "Paris is the capital of France.",
            "assessment": "true",
        }],
    }
    assert raw_queries(record) == ["Paris is in France.", "It lies on the Seine."]
    assert atomic_queries(record) == ["Paris is the capital of France."]


def test_raw_queries_use_final_output_block_not_instruction_marker() -> None:
    record = {
        "prompt": (
            "Judge the final <output> only.\n"
            "<output>Paris is in France. It lies on the Seine.</output>"
        ),
        "claims": [],
    }
    assert raw_queries(record) == ["Paris is in France.", "It lies on the Seine."]


def test_matched_shuffle_preserves_count_and_changes_dataset() -> None:
    rows = [
        {"dataset": "a", "index": 1, "real": [{"text": "A"}]},
        {"dataset": "b", "index": 2, "real": [{"text": "B"}]},
        {"dataset": "a", "index": 3, "real": []},
    ]
    add_matched_shuffle(rows, real_field="real", shuffled_field="shuffled")
    assert rows[0]["shuffled"] == [{"text": "B"}]
    assert rows[1]["shuffled"] == [{"text": "A"}]
    assert rows[2]["shuffled"] == []


def test_matched_shuffle_can_compose_two_passage_control() -> None:
    rows = [
        {"dataset": "a", "index": 1, "real": [{"text": "A1"}, {"text": "A2"}]},
        {"dataset": "b", "index": 2, "real": [{"text": "B"}]},
        {"dataset": "c", "index": 3, "real": [{"text": "C"}]},
    ]
    add_matched_shuffle(rows, real_field="real", shuffled_field="shuffled")
    assert rows[0]["shuffled"] == [{"text": "B"}, {"text": "C"}]


def test_claim_parser_requires_explicit_lines_and_deduplicates() -> None:
    assert parse_claims(
        "Here are the facts:\n"
        "CLAIM: Apollo 11 landed on the Moon in 1969.\n"
        "CLAIM: Apollo 11 landed on the Moon in 1969.\n"
        "CLAIM: Neil Armstrong commanded Apollo 11."
    ) == [
        "Apollo 11 landed on the Moon in 1969.",
        "Neil Armstrong commanded Apollo 11.",
    ]
    assert parse_claims("CLAIM: NONE") == []
