import json
from pathlib import Path
import sys

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.train_student_sft import (
    CompletionOnlyCollator,
    LABEL_ONLY_TARGET_KEY,
    apply_label_only_manifest,
    has_reasoning_block,
    load_record_sources,
    load_records,
    order_records_for_grouped_pair_batches,
    order_records_for_paired_batches,
    pairwise_logistic_loss,
    select_rating_uncertainty_fraction,
    select_rating_uncertainty_with_certain_anchors,
    select_records_from_manifest,
    select_stratified_fraction,
    should_drop_reasoning,
    soft_binary_distillation_loss,
    strip_reasoning_block,
    student_prompt_with_reasoning_dropout,
    tokenize_record,
    training_warmup_steps,
    validate_paired_batch_size,
    validate_trainable_lora_layout,
)
from experiments.qwen_grpo_lora.run_qwen_grpo_lora import (
    MuonAdamW,
    muon_adamw_param_groups,
)


class FakeParameter:
    def __init__(self, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def named_parameters(self):
        return [(name, FakeParameter()) for name in self.names]


def test_canonical_lora_layout_requires_language_model_tree() -> None:
    canonical = (
        "base_model.model.model.language_model.layers.0.self_attn."
        "q_proj.lora_A.default.weight"
    )
    assert validate_trainable_lora_layout(
        FakeModel([canonical]), "image_text_to_text"
    ) == [canonical]

    legacy = (
        "base_model.model.model.layers.0.self_attn."
        "q_proj.lora_A.default.weight"
    )
    try:
        validate_trainable_lora_layout(FakeModel([legacy]), "image_text_to_text")
    except RuntimeError as error:
        assert "non-language-model LoRA" in str(error)
    else:
        raise AssertionError("legacy text-only LoRA keys must fail canonical training")


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


def test_apply_label_only_manifest_marks_exact_rows_and_verifies_labels(tmp_path) -> None:
    records = [
        {"dataset": "dataset", "index": 1, "label": 0},
        {"dataset": "dataset", "index": 2, "label": 1},
    ]
    manifest = tmp_path / "label-only.jsonl"
    manifest.write_text(
        json.dumps({"dataset": "dataset", "index": 2, "label": 1}) + "\n"
    )

    marked = apply_label_only_manifest(records, manifest)

    assert marked[0][LABEL_ONLY_TARGET_KEY] is False
    assert marked[1][LABEL_ONLY_TARGET_KEY] is True
    assert LABEL_ONLY_TARGET_KEY not in records[0]

    manifest.write_text(
        json.dumps({"dataset": "dataset", "index": 2, "label": 0}) + "\n"
    )
    try:
        apply_label_only_manifest(records, manifest)
    except ValueError as error:
        assert "label mismatch" in str(error)
    else:
        raise AssertionError("label-only manifest mismatch should fail")


def test_ratio_warmup_uses_transformers_v5_warmup_steps_argument(tmp_path) -> None:
    from transformers import TrainingArguments

    warmup = training_warmup_steps(OmegaConf.create({"warmup_ratio": 0.03}))
    arguments = TrainingArguments(
        output_dir=(tmp_path / "trainer").as_posix(),
        warmup_steps=warmup,
    )

    assert warmup == 0.03
    assert arguments.get_warmup_steps(90) == 3


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


def test_direct_target_uses_rendered_prompt_without_teacher_summary() -> None:
    record = {
        "dataset": "dev-varied-deception-model",
        "index": 7,
        "label": 1,
        "student_prompt": "JUDGE\n\n<context>Evidence</context>",
        "student_target": (
            "<reasoning_summary>Teacher trace</reasoning_summary>\nPrediction:1"
        ),
    }

    tokenized = tokenize_record(
        record,
        FakeTokenizer(),
        1000,
        include_direct_target=True,
        dataset_id=3,
    )
    direct = "".join(chr(token) for token in tokenized["direct_input_ids"])

    assert direct.endswith("Prediction:")
    assert "Teacher trace" not in direct
    assert tokenized["binary_label"] == 1
    assert tokenized["dataset_id"] == 3


def test_paired_order_puts_opposite_labels_from_same_dataset_together() -> None:
    records = [
        {"dataset": dataset, "index": f"{dataset}-{label}-{index}", "label": label}
        for dataset in ("a", "b")
        for label in (0, 1)
        for index in range(3)
    ]

    ordered = order_records_for_paired_batches(records, seed=4)

    assert len(ordered) == len(records)
    assert {record["index"] for record in ordered} == {
        record["index"] for record in records
    }
    for offset in range(0, len(ordered), 2):
        first, second = ordered[offset:offset + 2]
        assert first["dataset"] == second["dataset"]
        assert first["label"] != second["label"]


def test_paired_batch_size_accepts_throughput_batch_and_rejects_odd() -> None:
    validate_paired_batch_size(2)
    validate_paired_batch_size(8)

    with pytest.raises(ValueError, match="positive even"):
        validate_paired_batch_size(7)


def test_grouped_pair_batches_use_four_by_four_same_dataset_blocks() -> None:
    records = [
        {"dataset": dataset, "index": f"{dataset}-{label}-{index}", "label": label}
        for dataset in ("a", "b")
        for label in (0, 1)
        for index in range(8)
    ]

    ordered = order_records_for_grouped_pair_batches(
        records,
        seed=9,
        batch_size=8,
    )

    assert len(ordered) == len(records)
    assert {record["index"] for record in ordered} == {
        record["index"] for record in records
    }
    for offset in range(0, len(ordered), 8):
        batch = ordered[offset:offset + 8]
        assert len({record["dataset"] for record in batch}) == 1
        assert [record["label"] for record in batch].count(0) == 4
        assert [record["label"] for record in batch].count(1) == 4


def test_pairwise_logistic_loss_rewards_correct_within_dataset_order() -> None:
    labels = torch.tensor([0, 1, 0, 1])
    dataset_ids = torch.tensor([0, 0, 1, 1])
    correct = pairwise_logistic_loss(
        torch.tensor([-2.0, 2.0, -1.0, 1.0]),
        labels,
        dataset_ids,
    )
    reversed_order = pairwise_logistic_loss(
        torch.tensor([2.0, -2.0, 1.0, -1.0]),
        labels,
        dataset_ids,
    )

    assert correct < reversed_order


def test_binary_soft_bce_identity_matches_original_probability_loss() -> None:
    logits = torch.tensor([[0.0, -2.0], [0.0, 3.0]])
    targets = torch.tensor([0.2, 0.8])

    actual = soft_binary_distillation_loss(logits, targets)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1] - logits[:, 0],
        targets,
    )

    assert torch.equal(actual, expected)


def test_binary_soft_bce_can_standardize_teacher_log_odds() -> None:
    targets = torch.sigmoid(torch.tensor([-6.0, 2.0]))
    centered_targets = torch.tensor([-1.0, 1.0])
    logits = torch.stack((torch.zeros(2), centered_targets), dim=1)

    loss = soft_binary_distillation_loss(
        logits,
        targets,
        loss_type="bce",
        target_logit_center=-2.0,
        target_logit_scale=4.0,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        centered_targets,
        torch.sigmoid(centered_targets),
    )

    assert torch.allclose(loss, expected)


def test_binary_soft_huber_regresses_standardized_teacher_margin() -> None:
    targets = torch.sigmoid(torch.tensor([-6.0, 2.0]))
    matched_margins = torch.tensor([-1.0, 1.0])
    matched_logits = torch.stack((torch.zeros(2), matched_margins), dim=1)
    shifted_logits = torch.stack((torch.zeros(2), matched_margins + 0.5), dim=1)

    matched = soft_binary_distillation_loss(
        matched_logits,
        targets,
        loss_type="huber",
        target_logit_center=-2.0,
        target_logit_scale=4.0,
        huber_delta=1.0,
    )
    shifted = soft_binary_distillation_loss(
        shifted_logits,
        targets,
        loss_type="huber",
        target_logit_center=-2.0,
        target_logit_scale=4.0,
        huber_delta=1.0,
    )

    assert torch.isclose(matched, torch.tensor(0.0), atol=1e-7)
    assert shifted > matched


def test_completion_collator_pads_direct_targets() -> None:
    batch = CompletionOnlyCollator(pad_token_id=0)([
        {
            "input_ids": [1, 2],
            "labels": [-100, 2],
            "direct_input_ids": [3, 4, 5],
            "binary_label": 0,
            "dataset_id": 7,
        },
        {
            "input_ids": [6],
            "labels": [6],
            "direct_input_ids": [8],
            "binary_label": 1,
            "dataset_id": 7,
        },
    ])

    assert batch["direct_input_ids"].tolist() == [[3, 4, 5], [8, 0, 0]]
    assert batch["direct_attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["binary_labels"].tolist() == [0, 1]


def test_label_only_manifest_row_omits_the_cached_teacher_summary() -> None:
    record = {
        "index": 8,
        "label": 1,
        "student_prompt": "TEACHER PROMPT\n\n<context>Evidence</context>",
        "student_target": (
            "<reasoning_summary>Contradictory teacher text</reasoning_summary>\n"
            "Prediction:1"
        ),
        LABEL_ONLY_TARGET_KEY: True,
    }

    tokenized = tokenize_record(
        record,
        FakeTokenizer(),
        1000,
        target_mode="teacher",
    )
    decoded = "".join(chr(token) for token in tokenized["input_ids"])
    supervised = "".join(
        chr(token)
        for token, label in zip(
            tokenized["input_ids"], tokenized["labels"], strict=True
        )
        if label != -100
    )

    assert "Contradictory teacher text" not in decoded
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
