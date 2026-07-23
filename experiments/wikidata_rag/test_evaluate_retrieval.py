from experiments.wikidata_rag.evaluate_retrieval import (
    datavalue,
    extract_conversation,
    fetch_entities,
    score_record,
    search_queries,
    tokens,
)
import requests


class FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = iter(payloads)

    def get(self, *args, **kwargs):
        payload = next(self.payloads)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload)


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


def test_fetch_entities_retries_structured_maxlag(monkeypatch) -> None:
    monkeypatch.setattr("experiments.wikidata_rag.evaluate_retrieval.time.sleep", lambda _: None)
    session = FakeSession([
        {"error": {"code": "maxlag", "info": "lagged"}},
        {"entities": {"Q42": {"id": "Q42"}}},
    ])
    assert fetch_entities(session, ["Q42"], 0) == {"Q42": {"id": "Q42"}}


def test_fetch_entities_retries_connection_error(monkeypatch) -> None:
    monkeypatch.setattr("experiments.wikidata_rag.evaluate_retrieval.time.sleep", lambda _: None)
    session = FakeSession([
        requests.ConnectionError("temporary DNS failure"),
        {"entities": {"Q42": {"id": "Q42"}}},
    ])
    assert fetch_entities(session, ["Q42"], 0) == {"Q42": {"id": "Q42"}}
