from experiments.wikidata_rag.evaluate_retrieval import (
    datavalue,
    extract_conversation,
    score_record,
    search_queries,
    tokens,
)


def test_extract_conversation() -> None:
    prompt = "instructions\n<context>\nUSER: question\nASSISTANT: answer\n</context>\n<output>x</output>"
    assert extract_conversation(prompt) == "USER: question\nASSISTANT: answer"


def test_datavalue_formats_entity_time_and_quantity() -> None:
    assert datavalue({"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q42"}}}) == ("entity", "Q42")
    assert datavalue({"datavalue": {"type": "time", "value": {"time": "+1952-03-11T00:00:00Z", "precision": 11}}}) == ("literal", "1952-03-11")
    assert datavalue({"datavalue": {"type": "quantity", "value": {"amount": "+8848"}}}) == ("literal", "8848")


def test_novel_target_recall_excludes_conversation_words() -> None:
    record = {
        "conversation": "The event was held in Tokyo.",
        "reasoning_summary": "The event was held in Rio de Janeiro, not Tokyo.",
    }
    score = score_record(record, "location: Rio de Janeiro")
    assert score["novel_target_recall"] == 1.0
    assert "the" not in tokens(record["reasoning_summary"])


def test_search_queries_separate_final_user_and_assistant() -> None:
    conversation = (
        "SYSTEM: Ignore this.\n\nUSER: Who founded the French Foreign Legion?\n\n"
        "ASSISTANT: The British Foreign Legion was founded in 1831."
    )
    queries = search_queries(conversation)
    assert queries[:2] == [
        "Who founded the French Foreign Legion?",
        "The British Foreign Legion was founded in 1831.",
    ]
    assert "USER:" not in " ".join(queries)
