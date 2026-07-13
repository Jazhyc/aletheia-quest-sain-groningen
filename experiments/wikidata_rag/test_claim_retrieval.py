import sqlite3

from experiments.wikidata_rag.claim_retrieval import (
    extract_claim_query,
    gate_entity,
    retrieve_claim_evidence,
    temporally_compatible,
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


def test_extracts_new_border_relation() -> None:
    claim = extract_claim_query(
        "USER: Which countries border Tunisia?\n\nASSISTANT: Algeria and Libya border Tunisia."
    )

    assert "border" in claim.relations
    assert claim.predicates == ("shares border with",)


def test_extracts_new_screenwriter_relation() -> None:
    claim = extract_claim_query(
        "USER: Who wrote the screenplay for Chinatown?\n\nASSISTANT: Robert Towne wrote it."
    )

    assert "screenwriter" in claim.relations
    assert "screenwriter" in claim.predicates


def test_winner_question_does_not_retrieve_generic_event_metadata() -> None:
    claim = extract_claim_query(
        "USER: Which notable leader won the 2009 Nobel Peace Prize?\n\n"
        "ASSISTANT: Barack Obama won it."
    )

    assert "event" in claim.relations
    assert claim.predicates == ("winner",)
    assert not temporally_compatible(
        "winner: European Union [point in time: 2012]", claim
    )
    assert temporally_compatible(
        "winner: Barack Obama [point in time: 2009]", claim
    )


def test_year_qualified_world_cup_prefers_edition_subject() -> None:
    claim = extract_claim_query(
        "USER: Which nation's football team won the FIFA World Cup in 2006?\n\n"
        "ASSISTANT: France won."
    )

    assert claim.subjects[0] == "2006 FIFA World Cup"
    assert claim.predicates == ("winner",)

    possessive = extract_claim_query(
        "USER: Which country's football team won the 1986 FIFA World Cup?\n\n"
        "ASSISTANT: Argentina won."
    )
    assert possessive.subjects[0] == "1986 FIFA World Cup"
    assert possessive.predicates == ("winner",)


def test_non_answerable_question_slots_abstain() -> None:
    questions = (
        "What battleship was sunk in Havana harbor in 1898?",
        "Which river forms the border between Zimbabwe and South Africa?",
        "Ernest Borgnine won an Oscar for which film?",
        "Who was prime minister during the reign of which monarch?",
        "Which sport was played at Wimbledon prior to lawn tennis?",
    )

    for question in questions:
        claim = extract_claim_query(f"USER: {question}\n\nASSISTANT: Example.")
        assert claim.predicates == (), question


def test_saint_alias_and_burial_question_select_direct_fact() -> None:
    claim = extract_claim_query(
        "USER: St George is England's patron saint, but in which country is his tomb?\n\n"
        "ASSISTANT: His tomb is in Lod."
    )
    saint = {
        "qid": "Q48438", "label": "Saint George", "aliases": ["St George"],
        "description": "Christian saint and martyr",
        "facts": ["instance of: human", "place of burial: Lod"],
    }
    locality = {
        "qid": "Q2", "label": "St. George", "aliases": [],
        "description": "community in the United States",
        "facts": ["country: United States"],
    }

    assert gate_entity(saint, claim)["facts"] == ["place of burial: Lod"]
    assert gate_entity(locality, claim) is None


def test_local_name_uses_matched_alias_as_direct_evidence() -> None:
    claim = extract_claim_query(
        "USER: Kerkyra is the local name for what island?\n\nASSISTANT: Corfu."
    )
    entity = {
        "qid": "Q205832", "label": "Corfu", "aliases": ["Kerkyra"],
        "description": "Greek island", "facts": ["instance of: island"],
    }

    assert gate_entity(entity, claim)["facts"] == ["alias: Kerkyra"]
    city = {
        "qid": "Q2", "label": "Corfu", "aliases": ["Kerkyra"],
        "description": "capital of the Greek island of Corfu",
        "facts": ["instance of: city", "official name: Κέρκυρα"],
    }
    assert gate_entity(city, claim) is None


def test_modern_country_discards_historical_scoped_values() -> None:
    claim = extract_claim_query(
        "USER: Carthage is in which modern-day country?\n\nASSISTANT: Tunisia."
    )
    entity = {
        "qid": "Q6343", "label": "Carthage", "aliases": [],
        "description": "ancient city", "facts": [
            "country: Tunisia",
            "country: Roman Empire [end time: 0395]",
            "located in: Exarchate of Africa",
        ],
    }

    assert gate_entity(entity, claim)["facts"] == ["country: Tunisia"]


def test_country_slot_prefers_direct_country_over_administration() -> None:
    claim = extract_claim_query(
        "USER: The Galapagos Islands belong to which country?\n\nASSISTANT: Ecuador."
    )
    entity = {
        "qid": "Q38095", "label": "Galapagos Islands", "aliases": [],
        "description": "archipelago", "facts": [
            "country: Ecuador", "located in: Galápagos Province",
        ],
    }

    assert gate_entity(entity, claim)["facts"] == ["country: Ecuador"]


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
