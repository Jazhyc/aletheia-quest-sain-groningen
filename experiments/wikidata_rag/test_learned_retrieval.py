from experiments.wikidata_rag.train_predicate_router import oracle_predicate


def test_oracle_predicate_uses_direct_card_schema() -> None:
    assert oracle_predicate("genre: detective fiction") == "genre"
    assert oracle_predicate("date of birth: 1943-06-01") == "date of birth"
