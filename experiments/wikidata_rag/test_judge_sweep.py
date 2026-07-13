import json

import pandas as pd

from experiments.wikidata_rag.build_validation_cache import (
    add_shuffled_passages,
    entity_passages,
)
from experiments.wikidata_rag.evaluate_judge_sweep import (
    load_condition_references,
    paired_changes,
)
from experiments.wikidata_rag.build_teacher_cache import convert_record


def test_entity_passages_caps_facts() -> None:
    passages = entity_passages([{
        "label": "Example",
        "qid": "Q1",
        "description": "description",
        "facts": [f"fact {number}" for number in range(10)],
    }])

    assert passages[0]["title"] == "Example (Q1)"
    assert "fact 7" in passages[0]["text"]
    assert "fact 8" not in passages[0]["text"]


def test_shuffled_passages_are_bijective_and_not_self_assigned() -> None:
    records = [
        {
            "dataset": f"dataset-{number // 3}",
            "index": number % 3,
            "conversation": f"topic {number}",
            "real_passages": [{"title": f"entity-{number}", "text": f"fact-{number}"}],
        }
        for number in range(6)
    ]

    add_shuffled_passages(records, seed=42)

    assert all(row["shuffled_from"]["index"] != row["index"] for row in records)
    assert len({
        (row["shuffled_from"]["dataset"], row["shuffled_from"]["index"])
        for row in records
    }) == len(records)


def test_load_condition_references_reads_real_and_shuffled(tmp_path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({
        "dataset": "varied",
        "index": 3,
        "real_passages": [{"title": "Real", "text": "real fact"}],
        "shuffled_passages": [{"title": "Null", "text": "null fact"}],
    }) + "\n")

    references, keys = load_condition_references(path)

    assert keys == {("varied", 3)}
    assert "Real" in references["real"][("varied", 3)]
    assert "Null" in references["shuffled"][("varied", 3)]


def test_paired_changes_counts_recoveries_and_harms() -> None:
    frame = pd.DataFrame([
        {"dataset": "dev-varied-deception-x", "index": 1, "label": 1, "condition": "empty", "prediction": 0},
        {"dataset": "dev-varied-deception-x", "index": 1, "label": 1, "condition": "real", "prediction": 1},
        {"dataset": "dev-varied-deception-x", "index": 2, "label": 0, "condition": "empty", "prediction": 0},
        {"dataset": "dev-varied-deception-x", "index": 2, "label": 0, "condition": "real", "prediction": 1},
    ])

    changes = paired_changes(frame, "real")

    assert changes["deceptive_recoveries_fn_to_tp"] == 1
    assert changes["honest_harms_tn_to_fp"] == 1


def test_convert_record_preserves_keys_and_formats_entities() -> None:
    converted = convert_record({
        "dataset": "varied",
        "index": 9,
        "entities": [{
            "qid": "Q1",
            "label": "Entity",
            "description": "description",
            "facts": ["country: Example"],
        }],
    })

    assert converted["dataset"] == "varied"
    assert converted["index"] == 9
    assert converted["qids"] == ["Q1"]
    assert converted["passages"] == [{
        "title": "Entity (Q1)",
        "text": "description; country: Example",
    }]


def test_convert_record_accepts_preformatted_real_passages() -> None:
    passages = [{"title": "Entity (Q1)", "text": "fact"}]

    converted = convert_record({
        "dataset": "varied",
        "index": 10,
        "real_passages": passages,
        "qids": ["Q1"],
    })

    assert converted["passages"] == passages
    assert converted["qids"] == ["Q1"]
