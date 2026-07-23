from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.filter_material_evidence import (
    build_filtered_cache,
    filter_passages,
    focus_source,
    repeats_disputed_question_value,
    salient_values,
)


def passage(
    claim: str,
    source: str,
    *,
    claim_index: int = 0,
    relation: str = "DECISIVE_CONTRADICTION",
) -> dict:
    return {
        "title": "Example",
        "text": f"Claim: {claim}\nSource sentence: {source}",
        "claim_index": claim_index,
        "audit_relation": relation,
    }


def test_salient_values_normalizes_quantities_and_ordinals() -> None:
    assert salient_values("The second event drew 1,000 people (30%).") == {
        "ordinal:2", "cardinal:1000", "percent:30%",
    }


def test_question_value_guard_is_narrow_and_label_blind() -> None:
    assert repeats_disputed_question_value(
        "Was VAT introduced in France in 1954?",
        "VAT was introduced in France in 1954.",
        "France was the first country to implement VAT in 1958.",
    )
    assert repeats_disputed_question_value(
        "Beira is the second largest city in which country?",
        "Beira is the second largest city in Mozambique.",
        "Beira is the fourth-largest city in Mozambique.",
    )
    assert not repeats_disputed_question_value(
        "Who directed Good Morning, Vietnam?",
        "Good Morning, Vietnam was directed by Barry Levenson.",
        "Good Morning, Vietnam was directed by Barry Levinson.",
    )
    assert not repeats_disputed_question_value(
        "Who starred in the title role in the 2002 film Mr Deeds?",
        "Bill Murray starred in the title role in the 2002 film Mr Deeds.",
        "The film stars Adam Sandler in the title role.",
    )
    assert not repeats_disputed_question_value(
        "In which year was the penalty kick first introduced?",
        "The penalty kick was first introduced in 1895.",
        "The penalty kick was introduced in 1891.",
    )
    assert not repeats_disputed_question_value(
        "Which planet made a transit in 2012?",
        "The next Venus transit after 2012 will be in 2117.",
        "The next pair of Venus transits begins in 2117.",
    )
    assert not repeats_disputed_question_value(
        "Who became Confederate president in 1861?",
        "Alexander Stephens became Confederate president in 1861.",
        "Alexander Stephens was vice president. Congress assembled in 1865.",
    )
    assert repeats_disputed_question_value(
        "What was the three-headed dog called?",
        "Cerberus had three heads and a snake tail.",
        "This depiction shows Cerberus with two heads and a snake tail.",
    )


def test_focused_source_removes_unrelated_window_sentence() -> None:
    original = passage(
        "Manchester United played Red Star Belgrade.",
        "Manchester United played Red Star Belgrade in the European Cup. "
        "The crash killed 23 people, including eight players.",
        relation="DECISIVE_SUPPORT",
    )
    focused, changed = focus_source(original)
    assert changed
    assert "European Cup" in focused["text"]
    assert "killed 23" not in focused["text"]


def test_single_mode_keeps_lowest_surviving_claim() -> None:
    rows = [
        passage("The answer is Paris.", "Paris is the capital of France."),
        passage(
            "Paris has two million residents.",
            "Paris has more than two million residents.",
            claim_index=2,
            relation="DECISIVE_SUPPORT",
        ),
    ]
    filtered, stats = filter_passages(rows, "What is France's capital?", mode="premise_single")
    assert [row["claim_index"] for row in filtered] == [0]
    assert stats["dropped_later_claims"] == 1


def test_cache_rebuilds_cross_dataset_shuffled_controls() -> None:
    references = [
        {"dataset": "a", "index": 1, "real_passages": [passage("Paris is in France.", "Paris is in France.")]},
        {"dataset": "b", "index": 2, "real_passages": [passage("Rome is in Italy.", "Rome is in Italy.")]},
        {"dataset": "a", "index": 3, "real_passages": []},
    ]
    questions = [
        {"dataset": "a", "index": 1, "question": "Where is Paris?"},
        {"dataset": "b", "index": 2, "question": "Where is Rome?"},
    ]
    output, summary = build_filtered_cache(references, questions, mode="premise_guard")
    assert summary["active_rows"] == 2
    assert output[0]["shuffled_donor_dataset"] == "b"
    assert output[1]["shuffled_donor_dataset"] == "a"
    assert output[2]["shuffled_passages"] == []
