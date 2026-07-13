import sqlite3

from experiments.wikidata_rag.build_relation_index import PREDICATE_IDS
from experiments.wikidata_rag.relation_retrieval import (
    query_facts,
    render_fact,
    retrieve_office_overlap,
)


def relation_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE node(id INTEGER PRIMARY KEY,qid TEXT,label TEXT);
        CREATE TABLE fact(subject INTEGER,predicate INTEGER,object INTEGER,literal TEXT,
          start_year INTEGER,end_year INTEGER,point_year INTEGER,applies_to INTEGER,
          for_work INTEGER,extras TEXT);
    """)
    return connection


def test_structured_query_filters_incompatible_point_year() -> None:
    connection = relation_database()
    connection.executemany("INSERT INTO node VALUES (?,?,?)", [
        (1, "Qwork", "Example Prize"), (2, "Qold", "Old Winner"),
        (3, "Qnew", "New Winner"),
    ])
    connection.executemany("INSERT INTO fact VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (1, PREDICATE_IDS["winner"], 2, None, None, None, 2008, None, None, ""),
        (1, PREDICATE_IDS["winner"], 3, None, None, None, 2009, None, None, ""),
    ])
    facts = query_facts(
        connection, subject_ids=(1,), predicates=("winner",), years={2009}
    )
    assert [render_fact(row) for row in facts] == ["winner: New Winner [2009]"]


def test_constrained_two_hop_finds_overlapping_monarch() -> None:
    connection = relation_database()
    connection.executemany("INSERT INTO node VALUES (?,?,?)", [
        (1, "Qperson", "Neville Chamberlain"), (2, "Qpm", "Prime Minister of the United Kingdom"),
        (3, "Quk", "United Kingdom"), (4, "Qking", "King of the United Kingdom"),
        (5, "Qgeorge", "George VI"),
    ])
    connection.executemany("INSERT INTO fact VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (1, PREDICATE_IDS["position held"], 2, None, 1937, 1940, None, None, None, ""),
        (2, PREDICATE_IDS["applies to jurisdiction"], 3, None, None, None, None, None, None, ""),
        (4, PREDICATE_IDS["applies to jurisdiction"], 3, None, None, None, None, None, None, ""),
        (5, PREDICATE_IDS["position held"], 4, None, 1936, 1952, None, None, None, ""),
    ])
    result = retrieve_office_overlap(connection, (1,), {5})
    assert result is not None
    assert result["status"] == "support"
    assert "George VI" in result["facts"][1]
