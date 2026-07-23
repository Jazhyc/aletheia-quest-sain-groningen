from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.core import (
    aggregate_evidence,
    last_user_message,
    lexical_relevance,
    load_grounded_claims,
    quote_is_grounded,
    split_sentences,
    select_document_sentences,
    verdict_from_probabilities,
)


def test_exact_quote_grounding() -> None:
    assert quote_is_grounded("capital of France", "Paris is the capital of France.")
    assert not quote_is_grounded("France's capital", "Paris is the capital of France.")


def test_load_grounded_claims_discards_rewritten_quote() -> None:
    rows = [{
        "variant": "material_assessed",
        "dataset": "varied",
        "index": 1,
        "label": 1,
        "prompt": "<output>Paris is in Germany.</output>",
        "claims": [
            {"quote": "Paris is in Germany.", "proposition": "Paris is in Germany.", "assessment": "false"},
            {"quote": "Paris lies in Germany.", "proposition": "Paris is in Germany.", "assessment": "false"},
        ],
    }]
    claims = load_grounded_claims(rows)
    assert len(claims) == 1
    assert claims[0]["claim_index"] == 0


def test_last_user_message_ignores_system_and_assistant() -> None:
    prompt = """<context>
SYSTEM: rule
USER: first
ASSISTANT: answer
USER: final question?
ASSISTANT: response
</context>"""
    assert last_user_message(prompt) == "final question?"


def test_sentence_split_and_short_filter() -> None:
    assert split_sentences("Tiny. Paris is the capital and largest city of France. It has museums.") == [
        "Paris is the capital and largest city of France."
    ]


def test_sentence_split_removes_mediawiki_section_headers() -> None:
    assert split_sentences(
        "A sufficiently long opening sentence. == History == The subject began in 1974."
    ) == [
        "A sufficiently long opening sentence.",
        "The subject began in 1974.",
    ]


def test_full_document_selector_keeps_lead_and_exact_number() -> None:
    sentences = [
        "The subject is an organization with a long history.",
        "This general lead sentence supplies useful entity context.",
        "An unrelated event occurred in another country in 1999.",
        "The organization was founded in Paris in 1974.",
    ]
    selected = select_document_sentences(
        "The organization was founded in 1974.", sentences, limit=3
    )
    assert [index for index, _, _ in selected] == [0, 1, 3]
    assert lexical_relevance("founded in 1974", sentences[3]) > lexical_relevance(
        "founded in 1974", sentences[2]
    )


def test_neutral_and_polar_verdicts() -> None:
    assert verdict_from_probabilities(0.8, 0.1, 0.1) == "SUPPORTS"
    assert verdict_from_probabilities(0.1, 0.1, 0.8) == "REFUTES"
    assert verdict_from_probabilities(0.4, 0.5, 0.1) == "NOT_ENOUGH_INFO"


def test_aggregate_prefers_non_neutral_evidence() -> None:
    result = aggregate_evidence([
        {"text": "irrelevant", "entailment": 0.1, "neutral": 0.8, "contradiction": 0.1},
        {"text": "refuting", "entailment": 0.02, "neutral": 0.08, "contradiction": 0.9},
    ])
    assert result["verdict"] == "REFUTES"
    assert result["evidence"][0]["text"] == "refuting"
