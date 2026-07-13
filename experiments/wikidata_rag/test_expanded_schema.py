from experiments.wikidata_rag.build_broad_index import format_facts, referenced_qids
from experiments.wikidata_rag.evaluate_retrieval import entity_stub


def test_entity_stub_retains_expanded_property_and_qualifiers() -> None:
    entity = {
        "labels": {"en": {"value": "Example"}},
        "aliases": {"en": [{"value": f"alias {number}"} for number in range(15)]},
        "descriptions": {"en": {"value": "test entity"}},
        "claims": {
            "P47": [{
                "rank": "normal",
                "mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q2"}}},
                "qualifiers": {
                    "P580": [{"datavalue": {"type": "time", "value": {
                        "time": "+1990-01-01T00:00:00Z", "precision": 9,
                    }}}],
                    "P518": [{"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q3"}}}],
                },
            }],
        },
    }

    stub = entity_stub(entity)

    assert len(stub["aliases"]) == 12
    assert stub["claims"][0]["property"] == "shares border with"
    assert {row["property"] for row in stub["claims"][0]["qualifiers"]} == {
        "start time", "applies to part",
    }
    assert referenced_qids([stub]) == {"Q2": 1, "Q3": 1}
    assert format_facts(stub, {"Q2": "Neighbor", "Q3": "Northern section"}) == (
        "shares border with: Neighbor [start time: 1990, applies to part: Northern section]"
    )
