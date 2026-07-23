import json
from pathlib import Path
import sqlite3

from experiments.wikidata_rag.build_relation_index import PREDICATE_IDS, build


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_builds_forward_and_reverse_relation_indexes(tmp_path: Path) -> None:
    cards = tmp_path / "cards.jsonl"
    labels = tmp_path / "labels.jsonl"
    database = tmp_path / "relations.sqlite"
    write_jsonl(labels, [{"qid": "Q2", "label": "Italy"}])
    write_jsonl(cards, [{
        "qid": "Q1", "label": "2006 World Cup", "claims": [{
            "property": "winner", "kind": "entity", "value": "Q2",
            "qualifiers": [{
                "property": "point in time", "kind": "literal", "value": "2006",
            }],
        }],
    }])

    report = build(cards, labels, database)

    connection = sqlite3.connect(database)
    row = connection.execute(
        "SELECT s.qid, o.label, f.point_year FROM fact f "
        "JOIN node s ON s.id=f.subject JOIN node o ON o.id=f.object "
        "WHERE f.predicate=?",
        (PREDICATE_IDS["winner"],),
    ).fetchone()
    connection.close()
    assert row == ("Q1", "Italy", 2006)
    assert report["facts"] == 1
