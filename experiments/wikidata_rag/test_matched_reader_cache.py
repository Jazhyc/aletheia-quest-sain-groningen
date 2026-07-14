from __future__ import annotations

from experiments.wikidata_rag.build_matched_reader_cache import (
    REFERENCE_MARKER,
    build_variants,
    donor_keys,
)


def teacher(dataset: str, index: int, target: str) -> dict:
    return {
        "dataset": dataset,
        "index": index,
        "label": index % 2,
        "student_prompt": (
            f"judge\n\n<context>row {index}</context>"
            f"{REFERENCE_MARKER}old\n</reference_material>"
        ),
        "student_target": target,
        "parse_error": False,
        "label_match": True,
    }


def utility(dataset: str, index: int) -> dict:
    return {
        "dataset": dataset,
        "index": index,
        "label": index % 2,
        "candidates": [
            {
                "id": "helpful", "subject": "A", "fact": "country: B",
                "controlled_utility": 0.2, "semantic_label": "decisive",
            },
            {
                "id": "harmful", "subject": "C", "fact": "date: 1900",
                "controlled_utility": -0.3, "semantic_label": "irrelevant",
            },
        ],
    }


def test_donors_are_label_blind_and_cross_dataset() -> None:
    keys = [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    donors = donor_keys(keys)

    assert all(donor[0] != key[0] for key, donor in donors.items())


def test_build_variants_pairs_use_and_ignore_targets() -> None:
    keys = [("a", 0), ("b", 1)]
    baseline = {key: teacher(*key, target=f"base-{key[1]}") for key in keys}
    evidence = {key: teacher(*key, target=f"evidence-{key[1]}") for key in keys}
    references = {key: f"real-{key[1]}" for key in keys}
    utilities = {key: utility(*key) for key in keys}

    rows = build_variants(
        baseline, evidence, references, utilities, minimum_gain=0.01
    )
    first = [row for row in rows if row["dataset"] == "a"]
    by_variant = {row["evidence_variant"]: row for row in first}

    assert set(by_variant) == {
        "retrieved_real", "empty", "shuffled", "reader_harmful", "reader_helpful"
    }
    assert by_variant["retrieved_real"]["student_target"] == "evidence-0"
    assert by_variant["empty"]["student_target"] == "base-0"
    assert "No retrieved reference material" in by_variant["empty"]["student_prompt"]
    assert "real-1" in by_variant["shuffled"]["student_prompt"]
    assert "country: B" in by_variant["reader_helpful"]["student_prompt"]
    assert "date: 1900" in by_variant["reader_harmful"]["student_prompt"]


def test_build_variants_rejects_label_mismatch() -> None:
    keys = [("a", 0), ("b", 1)]
    baseline = {key: teacher(*key, target="base") for key in keys}
    evidence = {key: teacher(*key, target="evidence") for key in keys}
    references = {key: "real" for key in keys}
    utilities = {key: utility(*key) for key in keys}
    utilities[("a", 0)]["label"] = 1

    try:
        build_variants(baseline, evidence, references, utilities, minimum_gain=0.01)
    except ValueError as error:
        assert "label mismatch" in str(error)
    else:
        raise AssertionError("expected label mismatch to be rejected")
