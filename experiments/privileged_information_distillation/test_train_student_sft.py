import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.train_student_sft import (
    has_reasoning_block,
    load_record_sources,
    load_records,
    select_rating_uncertainty_fraction,
    select_rating_uncertainty_with_certain_anchors,
    select_records_from_manifest,
    select_stratified_fraction,
    should_drop_reasoning,
    strip_reasoning_block,
    student_prompt_with_reasoning_dropout,
    tokenize_record,
)
from experiments.qwen_grpo_lora.run_qwen_grpo_lora import (
    MuonAdamW,
    muon_adamw_param_groups,
)


def test_load_records_filters_by_dataset_name(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    records = [
        {
            "dataset": "dev-instructed-deception-model",
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:0",
        },
        {
            "dataset": "dev-varied-deception-model",
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:1",
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    selected = load_records(path, dataset_name_contains="varied-deception")

    assert [record["dataset"] for record in selected] == ["dev-varied-deception-model"]


def test_load_records_without_filter_preserves_all_usable_rows(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    record = {
        "dataset": "dev-instructed-deception-model",
        "parse_error": False,
        "label_match": True,
        "student_target": "Prediction:0",
    }
    path.write_text(json.dumps(record) + "\n")

    assert load_records(path) == [record]


def test_load_records_can_retain_blind_teacher_label_errors(tmp_path) -> None:
    record = {
        "dataset": "dev-varied-deception-model",
        "index": 1,
        "label": 1,
        "prediction": 0,
        "parse_error": False,
        "label_match": False,
        "student_target": "<reasoning_summary>Blind error.</reasoning_summary>Prediction:0",
    }
    path = tmp_path / "blind.jsonl"
    path.write_text(json.dumps(record) + "\n")

    assert load_records(path, require_label_match=False) == [record]
    try:
        load_records(path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("privileged mode must still reject label mismatches")


def test_select_records_from_manifest_uses_exact_shared_keys(tmp_path) -> None:
    records = [
        {"dataset": "dataset", "index": index, "label": index % 2}
        for index in range(4)
    ]
    manifest = tmp_path / "selection.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(records[index]) for index in (1, 3)
        )
        + "\n"
    )

    assert select_records_from_manifest(records, manifest) == [records[1], records[3]]


def test_select_records_from_manifest_rejects_unavailable_row(tmp_path) -> None:
    manifest = tmp_path / "selection.jsonl"
    manifest.write_text(
        json.dumps({"dataset": "dataset", "index": 9, "label": 1}) + "\n"
    )

    try:
        select_records_from_manifest([], manifest)
    except ValueError as error:
        assert "unavailable rows" in str(error)
    else:
        raise AssertionError("missing fixed selection should fail")


def test_load_record_sources_combines_disjoint_filtered_caches(tmp_path) -> None:
    full = tmp_path / "full.jsonl"
    trace = tmp_path / "trace.jsonl"
    full.write_text("\n".join(json.dumps(record) for record in [
        {
            "dataset": "dev-instructed-deception-model",
            "index": 1,
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:0",
        },
        {
            "dataset": "dev-varied-deception-model",
            "index": 2,
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:1",
        },
    ]) + "\n")
    trace.write_text(json.dumps({
        "dataset": "dev-varied-deception-model",
        "index": 2,
        "parse_error": False,
        "label_match": True,
        "student_target": "Prediction:1",
    }) + "\n")

    records = load_record_sources([
        (full, "instructed-deception"),
        (trace, "varied-deception"),
    ])

    assert [(record["dataset"], record["index"]) for record in records] == [
        ("dev-instructed-deception-model", 1),
        ("dev-varied-deception-model", 2),
    ]


def test_load_record_sources_rejects_overlapping_rows(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    path.write_text(json.dumps({
        "dataset": "dataset",
        "index": 4,
        "parse_error": False,
        "label_match": True,
        "student_target": "Prediction:0",
    }) + "\n")

    try:
        load_record_sources([(path, None), (path, None)])
    except ValueError as error:
        assert "duplicate teacher record" in str(error)
    else:
        raise AssertionError("overlapping teacher sources should fail")


def test_load_record_sources_accepts_distinct_named_variants(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    rows = [
        {
            "dataset": "dataset",
            "index": 4,
            "evidence_variant": variant,
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:0",
        }
        for variant in ("real", "empty")
    ]
    path.write_text("\n".join(json.dumps(record) for record in rows) + "\n")

    loaded = load_record_sources(
        [(path, None)], record_identity_field="evidence_variant"
    )

    assert [record["evidence_variant"] for record in loaded] == ["real", "empty"]


def test_load_record_sources_rejects_duplicate_named_variant(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    record = {
        "dataset": "dataset",
        "index": 4,
        "evidence_variant": "real",
        "parse_error": False,
        "label_match": True,
        "student_target": "Prediction:0",
    }
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")

    try:
        load_record_sources(
            [(path, None)], record_identity_field="evidence_variant"
        )
    except ValueError as error:
        assert "duplicate teacher record" in str(error)
    else:
        raise AssertionError("duplicate named variants should fail")


def test_load_record_sources_applies_fraction_per_source(tmp_path) -> None:
    full = tmp_path / "full.jsonl"
    rows = [
        {
            "dataset": "competition-varied",
            "index": index,
            "label": label,
            "parse_error": False,
            "label_match": True,
            "student_target": f"Prediction:{label}",
        }
        for label in (0, 1)
        for index in range(10)
    ]
    full.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected = load_record_sources([(full, None, 0.2, 7)])

    assert len(selected) == 4
    assert sum(record["label"] == 0 for record in selected) == 2
    assert sum(record["label"] == 1 for record in selected) == 2
    assert selected == load_record_sources([(full, None, 0.2, 7)])


def test_source_prompt_override_is_applied_only_to_its_source(tmp_path) -> None:
    full = tmp_path / "teacher.jsonl"
    rows = [
        {
            "dataset": f"dataset-{index}",
            "index": index,
            "label": index,
            "parse_error": False,
            "label_match": True,
            "student_prompt": "CACHED PROMPT\n\n<context>Evidence</context>",
            "student_target": f"Prediction:{index}",
        }
        for index in (0, 1)
    ]
    full.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected = load_record_sources([
        (full, "dataset-0", 1.0, 0, None),
        (full, "dataset-1", 1.0, 0, "SPECIALIST PROMPT"),
    ])
    ordinary = tokenize_record(selected[0], FakeTokenizer(), 1000)
    specialist = tokenize_record(selected[1], FakeTokenizer(), 1000)
    ordinary_text = "".join(chr(token) for token in ordinary["input_ids"])
    specialist_text = "".join(chr(token) for token in specialist["input_ids"])

    assert "CACHED PROMPT" in ordinary_text
    assert "SPECIALIST PROMPT\n\n<context>Evidence</context>" in specialist_text
    assert "CACHED PROMPT" not in specialist_text


def test_select_stratified_fraction_balances_dataset_and_label() -> None:
    records = [
        {"dataset": dataset, "label": label, "index": index}
        for dataset in ("dataset-a", "dataset-b")
        for label in (0, 1)
        for index in range(10)
    ]

    selected = select_stratified_fraction(records, 0.2, seed=7)

    assert len(selected) == 8
    assert {
        (dataset, label): sum(
            record["dataset"] == dataset and record["label"] == label
            for record in selected
        )
        for dataset in ("dataset-a", "dataset-b")
        for label in (0, 1)
    } == {
        ("dataset-a", 0): 2,
        ("dataset-a", 1): 2,
        ("dataset-b", 0): 2,
        ("dataset-b", 1): 2,
    }
    assert selected == select_stratified_fraction(records, 0.2, seed=7)
    assert selected != select_stratified_fraction(records, 0.2, seed=8)


def test_select_stratified_fraction_preserves_small_strata_and_full_data() -> None:
    records = [
        {"dataset": "dataset", "label": 0, "index": 1},
        {"dataset": "dataset", "label": 1, "index": 2},
    ]

    assert select_stratified_fraction(records, 0.05, seed=0) == records
    assert select_stratified_fraction(records, 1.0, seed=0) == records


def test_select_stratified_fraction_rejects_invalid_fraction() -> None:
    for fraction in (0.0, -0.1, 1.1):
        try:
            select_stratified_fraction([], fraction, seed=0)
        except ValueError as error:
            assert "train_fraction" in str(error)
        else:
            raise AssertionError(f"fraction={fraction} should fail")


def test_select_rating_uncertainty_fraction_is_midpoint_focused_and_balanced() -> None:
    records = [
        {
            "dataset": dataset,
            "label": label,
            "index": label * 100 + rating * 10 + duplicate,
            "rating": rating,
        }
        for dataset in ("dataset-a", "dataset-b")
        for label in (0, 1)
        for rating in (1, 2, 3, 4, 5, 6, 7)
        for duplicate in range(2)
    ]

    selected = select_rating_uncertainty_fraction(records, 0.25, seed=7)

    assert len(selected) == 16
    assert all(record["rating"] in (3, 4, 5) for record in selected)
    assert {
        (dataset, label): sum(
            record["dataset"] == dataset and record["label"] == label
            for record in selected
        )
        for dataset in ("dataset-a", "dataset-b")
        for label in (0, 1)
    } == {
        ("dataset-a", 0): 4,
        ("dataset-a", 1): 4,
        ("dataset-b", 0): 4,
        ("dataset-b", 1): 4,
    }
    assert selected == select_rating_uncertainty_fraction(records, 0.25, seed=7)


def test_select_rating_uncertainty_fraction_rejects_missing_rating() -> None:
    try:
        select_rating_uncertainty_fraction(
            [{"dataset": "dataset", "label": 0, "index": 1}],
            0.1,
            seed=0,
        )
    except ValueError as error:
        assert "integer ratings 1--7" in str(error)
    else:
        raise AssertionError("rating-focused selection should require ratings")


def test_select_rating_uncertainty_with_certain_anchors_balances_both_ends() -> None:
    records = [
        {
            "dataset": "dataset",
            "label": label,
            "index": label * 100 + rating * 10 + duplicate,
            "rating": rating,
        }
        for label in (0, 1)
        for rating in (1, 2, 3, 4, 5, 6, 7)
        for duplicate in range(2)
    ]

    selected = select_rating_uncertainty_with_certain_anchors(
        records, 0.25, seed=11
    )

    assert len(selected) == 16
    for label in (0, 1):
        ratings = [r["rating"] for r in selected if r["label"] == label]
        assert len(ratings) == 8
        assert sum(rating in (3, 4, 5) for rating in ratings) == 4
        assert sum(rating in (1, 7) for rating in ratings) == 4


def test_select_rating_uncertainty_with_certain_anchors_rejects_large_fraction() -> None:
    try:
        select_rating_uncertainty_with_certain_anchors([], 0.6, seed=0)
    except ValueError as error:
        assert "anchor fraction" in str(error)
    else:
        raise AssertionError("overlapping uncertainty/anchor fractions should fail")


def test_reasoning_block_helpers_remove_only_rendered_suffix() -> None:
    prompt = (
        "Mention <assistant_reasoning> in instructions.\n\n"
        "<output>Answer</output>\n\n"
        "<assistant_reasoning>\nprivate trace\n</assistant_reasoning>"
    )

    assert has_reasoning_block(prompt)
    assert strip_reasoning_block(prompt) == (
        "Mention <assistant_reasoning> in instructions.\n\n<output>Answer</output>"
    )


def test_reasoning_dropout_is_stable_by_dataset_and_index() -> None:
    record = {
        "dataset": "dev-varied-deception-model",
        "index": 17,
        "student_prompt": "Prompt\n\n<assistant_reasoning>\ntrace\n</assistant_reasoning>",
    }

    first = should_drop_reasoning(record, 0.5, seed=3)
    assert should_drop_reasoning(record, 0.5, seed=3) is first
    assert not should_drop_reasoning(record, 0.0, seed=3)
    assert should_drop_reasoning(record, 1.0, seed=3)


def test_student_prompt_dropout_ignores_rows_without_trace() -> None:
    record = {
        "dataset": "dev-instructed-deception-model",
        "index": 5,
        "student_prompt": "Prompt without trace",
    }

    prompt, dropped = student_prompt_with_reasoning_dropout(record, 1.0, seed=0)

    assert prompt == record["student_prompt"]
    assert dropped is False


def test_muon_groups_and_updates_only_trainable_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    model[1].weight.requires_grad_(False)
    groups = muon_adamw_param_groups(model, weight_decay=0.0)

    algorithms = {group["algorithm"] for group in groups}
    assert algorithms == {"muon", "adamw"}
    assert all(model[1].weight is not param for group in groups for param in group["params"])

    optimizer = MuonAdamW(groups, lr=1e-6, muon_lr=3e-5)
    before = model[0].weight.detach().clone()
    model[0](torch.ones(1, 3)).sum().backward()
    optimizer.step()

    assert not torch.equal(before, model[0].weight)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_prediction_only_target_replaces_cached_reasoning_prompt() -> None:
    record = {
        "index": 7,
        "label": 1,
        "student_prompt": "OLD REASONING PROMPT\n\n<context>Evidence</context>",
        "student_target": "<reasoning_summary>Teacher trace</reasoning_summary>\nPrediction:1",
    }

    tokenized = tokenize_record(
        record,
        FakeTokenizer(),
        1000,
        prompt_template="NEW BINARY PROMPT",
        target_mode="prediction_only",
    )
    decoded = "".join(chr(token) for token in tokenized["input_ids"])
    supervised = "".join(
        chr(token)
        for token, label in zip(tokenized["input_ids"], tokenized["labels"], strict=True)
        if label != -100
    )

    assert "NEW BINARY PROMPT\n\n<context>Evidence</context>" in decoded
    assert "OLD REASONING PROMPT" not in decoded
    assert supervised == "Prediction:1<eos>"


def test_trace_dropout_selects_the_no_reasoning_prompt() -> None:
    record = {
        "dataset": "dev-varied-deception-model",
        "index": 9,
        "label": 1,
        "student_prompt": (
            "OLD TRACE PROMPT\n\n<context>Evidence</context>\n\n"
            "<assistant_reasoning>\nTrace\n</assistant_reasoning>"
        ),
        "student_target": "Prediction:1",
    }

    tokenized = tokenize_record(
        record,
        FakeTokenizer(),
        1000,
        prompt_template="NEW TRACE PROMPT",
        prompt_template_without_reasoning="NEW ORDINARY PROMPT",
        reasoning_dropout_probability=1.0,
    )
    decoded = "".join(chr(token) for token in tokenized["input_ids"])

    assert "NEW ORDINARY PROMPT\n\n<context>Evidence</context>" in decoded
    assert "NEW TRACE PROMPT" not in decoded
    assert "<assistant_reasoning>" not in decoded
