import sqlite3

from experiments.wikidata_rag.claim_retrieval import (
    extract_claim_query,
    gate_entity,
    retrieve_claim_evidence,
)


def test_extracts_location_subject_and_answer() -> None:
    claim = extract_claim_query(
        "USER: In which county is the town and fishing port of Padstow?\n\n"
        "ASSISTANT: Padstow is located in the county of Devon, in South West England."
    )

    assert "location" in claim.relations
    assert "located in" in claim.predicates
    assert any("Padstow" in subject for subject in claim.subjects)
    assert any("Devon" in value for value in claim.answer_values)


def test_extracts_director_relation_and_work_subject() -> None:
    claim = extract_claim_query(
        "USER: Who directed the 1956 film Moby Dick?\n\n"
        "ASSISTANT: The film was directed by John Huston."
    )

    assert claim.relations == ("director",)
    assert "director" in claim.predicates
    assert any("Moby Dick" in subject for subject in claim.subjects)


def test_quoted_work_is_prioritized_over_franchise_context() -> None:
    claim = extract_claim_query(
        "USER: Who sang the theme tune to the James Bond film 'The Spy Who Loved Me' in 1977?\n\n"
        "ASSISTANT: Carly Simon sang it."
    )

    assert claim.subjects[0] == "The Spy Who Loved Me"
    assert claim.relations == ("performer",)


def test_relative_where_does_not_trigger_location_relation() -> None:
    claim = extract_claim_query(
        "USER: What is the title of the film where three girls go missing?\n\n"
        "ASSISTANT: It is Picnic at Hanging Rock."
    )

    assert "location" not in claim.relations


def test_unsupported_directional_relation_abstains() -> None:
    claim = extract_claim_query(
        "USER: Which state lies immediately north of North Carolina?\n\n"
        "ASSISTANT: Virginia."
    )

    assert claim.predicates == ()


def test_gate_rejects_topical_entity_without_requested_relation() -> None:
    claim = extract_claim_query(
        "USER: In which county is Padstow?\n\nASSISTANT: Padstow is in Devon."
    )
    entity = {
        "qid": "Q1",
        "label": "Padstow",
        "aliases": [],
        "description": "town in England",
        "facts": ["instance of: town", "population: 3000"],
    }

    assert gate_entity(entity, claim) is None


def test_gate_keeps_only_relation_aligned_facts() -> None:
    claim = extract_claim_query(
        "USER: In which county is Padstow?\n\nASSISTANT: Padstow is in Devon."
    )
    entity = {
        "qid": "Q1",
        "label": "Padstow",
        "aliases": [],
        "description": "town in England",
        "facts": ["located in: Cornwall", "population: 3000", "country: United Kingdom"],
    }

    gated = gate_entity(entity, claim)

    assert gated is not None
    assert gated["facts"] == ["located in: Cornwall"]


def test_gate_rejects_wrong_entity_with_shared_topic() -> None:
    claim = extract_claim_query(
        "USER: Braeburn, Jazz, Gala, and Fuji are varieties of what?\n\n"
        "ASSISTANT: They are varieties of apples."
    )
    entity = {
        "qid": "Q43202",
        "label": "Apples",
        "aliases": [],
        "description": "village in Switzerland",
        "facts": ["instance of: village", "country: Switzerland"],
    }

    assert gate_entity(entity, claim) is None


def test_gate_rejects_same_name_entity_of_wrong_work_type() -> None:
    claim = extract_claim_query(
        "USER: The UK television series 'Bergerac' was set on which island?\n\n"
        "ASSISTANT: It was set on Guernsey."
    )
    entity = {
        "qid": "Q1",
        "label": "Bergerac",
        "aliases": [],
        "description": "French commune in Dordogne",
        "facts": ["located in: Dordogne"],
    }

    assert gate_entity(entity, claim) is None


def make_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE entity (qid TEXT, label TEXT, aliases TEXT, description TEXT, "
        "facts TEXT, source TEXT, popularity INTEGER)"
    )
    connection.execute(
        "CREATE VIRTUAL TABLE entity_fts USING fts5("
        "label, aliases, description, facts, content='entity', content_rowid='rowid')"
    )
    connection.execute(
        "INSERT INTO entity VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Q1", "Padstow", "", "town in Cornwall", "located in: Cornwall; country: United Kingdom", "test", 10),
    )
    connection.execute("INSERT INTO entity_fts(entity_fts) VALUES ('rebuild')")
    return connection


def test_retrieval_abstains_or_returns_decisive_relation_fact() -> None:
    connection = make_database()

    result = retrieve_claim_evidence(
        connection,
        "USER: In which county is Padstow?\n\nASSISTANT: Padstow is in Devon.",
    )

    assert result["abstain_reason"] is None
    assert result["entities"][0]["label"] == "Padstow"
    assert "located in: Cornwall" in result["entities"][0]["facts"]
