#!/usr/bin/env python3
"""SFT a Qwen LoRA on concise privileged-teacher reasoning summaries."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qwen_grpo_lora.run_qwen_grpo_lora import (
    MuonAdamW,
    muon_adamw_param_groups,
)


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


REASONING_BLOCK_START = "\n\n<assistant_reasoning>\n"
REASONING_BLOCK_END = "\n</assistant_reasoning>"
SOURCE_PROMPT_TEMPLATE_KEY = "_source_prompt_template"
LABEL_ONLY_TARGET_KEY = "_label_only_target"


def has_reasoning_block(prompt: str) -> bool:
    """Return whether a rendered student prompt ends in a reasoning field."""
    start = prompt.rfind(REASONING_BLOCK_START)
    return start >= 0 and prompt.rstrip().endswith(REASONING_BLOCK_END)


def strip_reasoning_block(prompt: str) -> str:
    """Remove the final rendered reasoning field without touching prompt prose."""
    start = prompt.rfind(REASONING_BLOCK_START)
    if start < 0 or not prompt.rstrip().endswith(REASONING_BLOCK_END):
        return prompt
    return prompt[:start]


def should_drop_reasoning(
    record: dict[str, Any],
    probability: float,
    seed: int,
) -> bool:
    """Choose a stable per-row trace-dropout mask independent of row order."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "student.reasoning_dropout_probability must be between 0 and 1"
        )
    if probability == 0.0:
        return False
    if probability == 1.0:
        return True
    key = f"{seed}\0{record.get('dataset', '')}\0{record.get('index', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return fraction < probability


def student_prompt_with_reasoning_dropout(
    record: dict[str, Any],
    probability: float,
    seed: int,
) -> tuple[str, bool]:
    """Return the cached prompt after deterministic student-side trace dropout."""
    prompt = str(record["student_prompt"])
    dropped = has_reasoning_block(prompt) and should_drop_reasoning(
        record, probability, seed
    )
    return (strip_reasoning_block(prompt), True) if dropped else (prompt, False)


def load_records(
    path: Path,
    dataset_name_contains: str | None = None,
    *,
    require_label_match: bool = True,
) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    usable = [
        record for record in records
        if not record.get("parse_error")
        and (not require_label_match or record.get("label_match"))
        and record.get("student_target")
        and (
            dataset_name_contains is None
            or dataset_name_contains in str(record.get("dataset", ""))
        )
    ]
    if not usable:
        raise RuntimeError(f"no usable teacher records in {path}")
    return usable


def select_records_from_manifest(
    records: list[dict[str, Any]], manifest_path: Path
) -> list[dict[str, Any]]:
    """Select an exact shared dataset/index set and verify its labels."""
    desired: dict[tuple[str, Any], int] = {}
    for line_number, line in enumerate(
        manifest_path.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["dataset"]), record["index"])
        if key in desired:
            raise ValueError(f"duplicate selection key at line {line_number}: {key}")
        desired[key] = int(record["label"])
    if not desired:
        raise ValueError(f"selection manifest is empty: {manifest_path}")

    available = {
        (str(record.get("dataset", "")), record.get("index")): record
        for record in records
    }
    missing = sorted(set(desired) - set(available))
    if missing:
        raise ValueError(
            f"selection manifest has {len(missing)} unavailable rows; first={missing[0]}"
        )
    for key, label in desired.items():
        if int(available[key]["label"]) != label:
            raise ValueError(
                f"selection manifest label mismatch for {key}: "
                f"{label} != {available[key]['label']}"
            )
    return [
        record
        for record in records
        if (str(record.get("dataset", "")), record.get("index")) in desired
    ]


def apply_label_only_manifest(
    records: list[dict[str, Any]], manifest_path: Path
) -> list[dict[str, Any]]:
    """Mark exact rows to retain only their authoritative binary target."""
    desired: dict[tuple[str, Any], int] = {}
    for line_number, line in enumerate(
        manifest_path.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue
        manifest_record = json.loads(line)
        key = (str(manifest_record["dataset"]), manifest_record["index"])
        if key in desired:
            raise ValueError(
                f"duplicate label-only key at line {line_number}: {key}"
            )
        desired[key] = int(manifest_record["label"])
    if not desired:
        raise ValueError(f"label-only manifest is empty: {manifest_path}")

    available = {
        (str(record.get("dataset", "")), record.get("index")): record
        for record in records
    }
    missing = sorted(set(desired) - set(available))
    if missing:
        raise ValueError(
            f"label-only manifest has {len(missing)} unavailable rows; "
            f"first={missing[0]}"
        )
    for key, label in desired.items():
        if int(available[key]["label"]) != label:
            raise ValueError(
                f"label-only manifest label mismatch for {key}: "
                f"{label} != {available[key]['label']}"
            )

    marked = []
    for record in records:
        selected_record = dict(record)
        key = (str(record.get("dataset", "")), record.get("index"))
        selected_record[LABEL_ONLY_TARGET_KEY] = key in desired
        marked.append(selected_record)
    return marked


def training_warmup_steps(training_cfg: DictConfig) -> float:
    """Return the v5 warmup argument while preserving ratio-based configs."""
    configured_steps = OmegaConf.select(training_cfg, "warmup_steps", default=None)
    if configured_steps is not None:
        return float(configured_steps)
    ratio = float(training_cfg.warmup_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("student.training.warmup_ratio must be in [0, 1)")
    # Transformers v5 interprets a warmup_steps float below one as a ratio.
    return ratio


def load_record_sources(
    sources: list[
        tuple[Path, str | None]
        | tuple[Path, str | None, float, int]
        | tuple[Path, str | None, float, int, str | None]
    ],
    *,
    require_label_match: bool = True,
    record_identity_field: str | None = None,
) -> list[dict[str, Any]]:
    """Load cache slices, optionally distinguishing intentional row variants."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, Any] | tuple[str, Any, str]] = set()
    for source in sources:
        if len(source) == 2:
            path, dataset_name_contains = source
            fraction, seed = 1.0, 0
            source_prompt_template = None
        elif len(source) == 4:
            path, dataset_name_contains, fraction, seed = source
            source_prompt_template = None
        elif len(source) == 5:
            path, dataset_name_contains, fraction, seed, source_prompt_template = source
        else:
            raise ValueError(f"invalid teacher source tuple length: {len(source)}")
        source_records = load_records(
            path,
            dataset_name_contains=dataset_name_contains,
            require_label_match=require_label_match,
        )
        source_records = select_stratified_fraction(
            source_records, float(fraction), int(seed)
        )
        print(
            f"teacher source path={path} filter={dataset_name_contains!r} "
            f"fraction={float(fraction)} seed={int(seed)} "
            f"prompt_override={source_prompt_template is not None} "
            f"selected={len(source_records)}"
        )
        for record in source_records:
            base_key = (str(record.get("dataset", "")), record.get("index"))
            if record_identity_field is None:
                key: tuple[str, Any] | tuple[str, Any, str] = base_key
            else:
                identity = record.get(record_identity_field)
                if identity is None or str(identity).strip() == "":
                    raise ValueError(
                        f"teacher record {base_key} is missing non-empty "
                        f"identity field {record_identity_field!r}"
                    )
                key = (*base_key, str(identity))
            if key in seen:
                raise ValueError(f"duplicate teacher record across sources: {key}")
            seen.add(key)
            selected_record = dict(record)
            if source_prompt_template is not None:
                selected_record[SOURCE_PROMPT_TEMPLATE_KEY] = str(
                    source_prompt_template
                )
            records.append(selected_record)
    if not records:
        raise RuntimeError("no usable records across teacher sources")
    return records


def select_stratified_fraction(
    records: list[dict[str, Any]],
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a stable fraction within every dataset/label stratum."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("student.train_fraction must be in (0, 1]")
    if fraction == 1.0:
        return list(records)

    strata: dict[
        tuple[str, int],
        list[tuple[bytes, tuple[str, int, Any]]],
    ] = {}
    for record in records:
        key = (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        stratum = key[:2]
        digest = hashlib.sha256(
            f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}".encode("utf-8")
        ).digest()
        strata.setdefault(stratum, []).append((digest, key))

    selected: set[tuple[str, int, Any]] = set()
    for candidates in strata.values():
        count = max(1, int(len(candidates) * fraction + 0.5))
        selected.update(key for _, key in sorted(candidates)[:count])
    return [
        record
        for record in records
        if (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        ) in selected
    ]


def select_rating_uncertainty_fraction(
    records: list[dict[str, Any]],
    fraction: float,
    seed: int,
    *,
    midpoint: int = 4,
) -> list[dict[str, Any]]:
    """Select the ratings nearest the neutral midpoint within each stratum."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("rating_uncertainty_fraction must be in (0, 1]")
    if not 1 <= midpoint <= 7:
        raise ValueError("rating uncertainty midpoint must be between 1 and 7")
    if fraction == 1.0:
        return list(records)

    strata: dict[
        tuple[str, int],
        list[tuple[int, bytes, tuple[str, int, Any]]],
    ] = {}
    for record in records:
        rating = record.get("rating")
        if not isinstance(rating, int) or not 1 <= rating <= 7:
            raise ValueError(
                "rating uncertainty selection requires integer ratings 1--7"
            )
        key = (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        digest = hashlib.sha256(
            f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}".encode("utf-8")
        ).digest()
        strata.setdefault(key[:2], []).append(
            (abs(rating - midpoint), digest, key)
        )

    selected: set[tuple[str, int, Any]] = set()
    for candidates in strata.values():
        count = max(1, int(len(candidates) * fraction + 0.5))
        selected.update(key for _, _, key in sorted(candidates)[:count])
    return [
        record
        for record in records
        if (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        ) in selected
    ]


def select_rating_uncertainty_with_certain_anchors(
    records: list[dict[str, Any]],
    fraction_each: float,
    seed: int,
    *,
    midpoint: int = 4,
) -> list[dict[str, Any]]:
    """Select equal midpoint-near and extreme sets within every stratum."""
    if not 0.0 < fraction_each <= 0.5:
        raise ValueError("rating anchor fraction must be in (0, 0.5]")
    if not 1 <= midpoint <= 7:
        raise ValueError("rating uncertainty midpoint must be between 1 and 7")

    strata: dict[
        tuple[str, int],
        list[tuple[int, bytes, tuple[str, int, Any]]],
    ] = {}
    for record in records:
        rating = record.get("rating")
        if not isinstance(rating, int) or not 1 <= rating <= 7:
            raise ValueError(
                "rating anchor selection requires integer ratings 1--7"
            )
        key = (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        digest = hashlib.sha256(
            f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}".encode("utf-8")
        ).digest()
        strata.setdefault(key[:2], []).append(
            (abs(rating - midpoint), digest, key)
        )

    selected: set[tuple[str, int, Any]] = set()
    for stratum, candidates in strata.items():
        count = max(1, int(len(candidates) * fraction_each + 0.5))
        if count * 2 > len(candidates):
            raise ValueError(
                f"rating anchor sets overlap in stratum={stratum!r}: "
                f"2 * {count} > {len(candidates)}"
            )
        uncertain = sorted(candidates)[:count]
        certain = sorted(candidates, key=lambda item: (-item[0], item[1]))[:count]
        selected.update(key for _, _, key in uncertain)
        selected.update(key for _, _, key in certain)
    return [
        record
        for record in records
        if (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        ) in selected
    ]


def tokenize_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    *,
    prompt_template: str | None = None,
    prompt_template_without_reasoning: str | None = None,
    target_mode: str = "teacher",
    reasoning_dropout_probability: float = 0.0,
    reasoning_dropout_seed: int = 0,
) -> dict[str, list[int]]:
    raw_prompt, _ = student_prompt_with_reasoning_dropout(
        record,
        reasoning_dropout_probability,
        reasoning_dropout_seed,
    )
    source_prompt_template = record.get(SOURCE_PROMPT_TEMPLATE_KEY)
    effective_prompt_template = (
        str(source_prompt_template)
        if source_prompt_template is not None
        else prompt_template
    )
    if effective_prompt_template is not None:
        _, separator, evidence = raw_prompt.partition("<context>")
        if not separator:
            raise ValueError(f"student prompt is missing <context> for index={record['index']}")
        selected_template = effective_prompt_template
        if (
            source_prompt_template is None
            and not has_reasoning_block(raw_prompt)
            and prompt_template_without_reasoning
        ):
            selected_template = prompt_template_without_reasoning
        raw_prompt = f"{selected_template}\n\n<context>{evidence}"
    effective_target_mode = (
        "prediction_only" if record.get(LABEL_ONLY_TARGET_KEY) else target_mode
    )
    if effective_target_mode == "teacher":
        target = record["student_target"]
    elif effective_target_mode == "prediction_only":
        target = f"Prediction:{int(record['label'])}"
    else:
        raise ValueError(f"unknown student.target_mode={effective_target_mode!r}")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(
        target + (tokenizer.eos_token or ""),
        add_special_tokens=False,
    )
    if len(target_ids) >= max_length:
        raise ValueError(f"target alone exceeds student.max_length for index={record['index']}")
    prompt_ids = prompt_ids[-(max_length - len(target_ids)):]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="privileged_information_distillation",
)
def main(cfg: DictConfig) -> None:
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    class MuonSFTTrainer(Trainer):
        """Trainer using Muon for 2D LoRA matrices and AdamW otherwise."""

        def create_optimizer(self) -> torch.optim.Optimizer:
            if self.optimizer is None:
                self.optimizer = MuonAdamW(
                    muon_adamw_param_groups(self.model, float(self.args.weight_decay)),
                    lr=float(self.args.learning_rate),
                    muon_lr=float(cfg.student.training.muon_learning_rate),
                    muon_momentum=float(cfg.student.training.muon_momentum),
                    muon_nesterov=bool(cfg.student.training.muon_nesterov),
                    muon_ns_steps=int(cfg.student.training.muon_ns_steps),
                )
            return self.optimizer

    root = Path(get_original_cwd()).resolve()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    teacher_sources = OmegaConf.select(cfg, "student.teacher_sources", default=None)
    if teacher_sources is None:
        artifact = Path(str(cfg.teacher.artifact))
        if not artifact.is_absolute():
            artifact = root / artifact
        dataset_name_contains = (
            None
            if cfg.student.dataset_name_contains is None
            else str(cfg.student.dataset_name_contains)
        )
        sources = [(artifact, dataset_name_contains, 1.0, int(cfg.seed))]
    else:
        sources = []
        for source in teacher_sources:
            artifact = Path(str(source.artifact))
            if not artifact.is_absolute():
                artifact = root / artifact
            source_filter = OmegaConf.select(
                source, "dataset_name_contains", default=None
            )
            source_fraction = float(OmegaConf.select(
                source, "train_fraction", default=1.0
            ))
            source_seed = int(OmegaConf.select(
                source, "train_fraction_seed", default=cfg.seed
            ))
            source_prompt = OmegaConf.select(source, "prompt", default=None)
            sources.append((
                artifact,
                None if source_filter is None else str(source_filter),
                source_fraction,
                source_seed,
                None if source_prompt is None else str(source_prompt),
            ))
        dataset_name_contains = "multi-source"
    require_teacher_label_match = bool(OmegaConf.select(
        cfg, "student.require_teacher_label_match", default=True
    ))
    record_identity_field_value = OmegaConf.select(
        cfg, "student.record_identity_field", default=None
    )
    record_identity_field = (
        None
        if record_identity_field_value is None
        else str(record_identity_field_value)
    )
    records = load_record_sources(
        sources,
        require_label_match=require_teacher_label_match,
        record_identity_field=record_identity_field,
    )
    train_fraction = float(
        OmegaConf.select(cfg, "student.train_fraction", default=1.0)
    )
    train_fraction_seed = int(
        OmegaConf.select(cfg, "student.train_fraction_seed", default=cfg.seed)
    )
    records_before_fraction = len(records)
    selection_manifest = OmegaConf.select(
        cfg, "student.selection_manifest", default=None
    )
    rating_uncertainty_fraction = OmegaConf.select(
        cfg, "student.rating_uncertainty_fraction", default=None
    )
    if selection_manifest is not None:
        if train_fraction != 1.0 or rating_uncertainty_fraction is not None:
            raise ValueError(
                "selection_manifest requires train_fraction=1.0 and no "
                "rating_uncertainty_fraction"
            )
        manifest_path = Path(str(selection_manifest))
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        records = select_records_from_manifest(records, manifest_path)
        selection_mode = "fixed_manifest"
    elif rating_uncertainty_fraction is None:
        records = select_stratified_fraction(
            records,
            train_fraction,
            train_fraction_seed,
        )
        selection_mode = "random_stratified"
    else:
        if train_fraction != 1.0:
            raise ValueError(
                "set student.train_fraction=1.0 when using "
                "student.rating_uncertainty_fraction"
            )
        rating_uncertainty_seed = int(OmegaConf.select(
            cfg,
            "student.rating_uncertainty_seed",
            default=train_fraction_seed,
        ))
        rating_balance_certain = bool(OmegaConf.select(
            cfg,
            "student.rating_balance_certain",
            default=False,
        ))
        if rating_balance_certain:
            records = select_rating_uncertainty_with_certain_anchors(
                records,
                float(rating_uncertainty_fraction),
                rating_uncertainty_seed,
            )
            selection_mode = "rating_uncertainty_certain_balanced"
        else:
            records = select_rating_uncertainty_fraction(
                records,
                float(rating_uncertainty_fraction),
                rating_uncertainty_seed,
            )
            selection_mode = "rating_uncertainty_stratified"
    if cfg.student.train_limit is not None:
        records = records[:int(cfg.student.train_limit)]
    label_only_manifest = OmegaConf.select(
        cfg, "student.label_only_manifest", default=None
    )
    if label_only_manifest is not None:
        label_only_path = Path(str(label_only_manifest))
        if not label_only_path.is_absolute():
            label_only_path = root / label_only_path
        records = apply_label_only_manifest(records, label_only_path)
    label_only_rows = sum(
        bool(record.get(LABEL_ONLY_TARGET_KEY)) for record in records
    )
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.student.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    reasoning_dropout_probability = float(
        OmegaConf.select(
            cfg, "student.reasoning_dropout_probability", default=0.0
        )
    )
    reasoning_dropout_seed = int(
        OmegaConf.select(cfg, "student.reasoning_dropout_seed", default=cfg.seed)
    )
    reasoning_rows = sum(
        has_reasoning_block(str(record["student_prompt"])) for record in records
    )
    reasoning_rows_dropped = sum(
        student_prompt_with_reasoning_dropout(
            record,
            reasoning_dropout_probability,
            reasoning_dropout_seed,
        )[1]
        for record in records
    )
    tokenized = [
        tokenize_record(
            record,
            tokenizer,
            int(cfg.student.max_length),
            prompt_template=(
                str(cfg.student.prompt)
                if (
                    str(cfg.student.target_mode) == "prediction_only"
                    or bool(OmegaConf.select(
                        cfg, "student.override_cached_prompt", default=False
                    ))
                )
                else None
            ),
            prompt_template_without_reasoning=(
                str(cfg.student.prompt_without_reasoning)
                if (
                    OmegaConf.select(
                        cfg, "student.prompt_without_reasoning", default=None
                    ) is not None
                    and (
                        str(cfg.student.target_mode) == "prediction_only"
                        or bool(OmegaConf.select(
                            cfg, "student.override_cached_prompt", default=False
                        ))
                    )
                )
                else None
            ),
            target_mode=str(cfg.student.target_mode),
            reasoning_dropout_probability=reasoning_dropout_probability,
            reasoning_dropout_seed=reasoning_dropout_seed,
        )
        for record in records
    ]
    dataset = Dataset.from_list(tokenized)
    print(
        f"training on {len(dataset)} parsed, label-consistent teacher targets "
        f"records_before_fraction={records_before_fraction} "
        f"train_fraction={train_fraction} "
        f"train_fraction_seed={train_fraction_seed} "
        f"selection_mode={selection_mode} "
        f"rating_uncertainty_fraction={rating_uncertainty_fraction} "
        f"selection_manifest={selection_manifest} "
        f"label_only_manifest={label_only_manifest} "
        f"label_only_rows={label_only_rows} "
        f"record_identity_field={record_identity_field!r} "
        f"require_teacher_label_match={require_teacher_label_match} "
        f"dataset_name_contains={dataset_name_contains!r} "
        f"reasoning_rows={reasoning_rows} "
        f"reasoning_rows_dropped={reasoning_rows_dropped} "
        f"reasoning_dropout_probability={reasoning_dropout_probability}"
    )
    rating_counts = Counter(
        record.get("rating") for record in records if record.get("rating") is not None
    )
    if rating_counts:
        print(f"selected_rating_counts={dict(sorted(rating_counts.items()))}")

    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.student.model),
        torch_dtype=torch.bfloat16,
    )
    init_adapter_value = OmegaConf.select(
        cfg, "student.init_adapter", default=None
    )
    if init_adapter_value is None:
        model = get_peft_model(model, LoraConfig(
            r=int(cfg.student.lora.r),
            lora_alpha=int(cfg.student.lora.alpha),
            lora_dropout=float(cfg.student.lora.dropout),
            target_modules=list(cfg.student.lora.target_modules),
            task_type="CAUSAL_LM",
        ))
    else:
        init_adapter = Path(str(init_adapter_value))
        if not init_adapter.is_absolute():
            init_adapter = root / init_adapter
        if not (init_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing initial adapter: {init_adapter}")
        print(f"loading trainable initial adapter from {init_adapter}")
        model = PeftModel.from_pretrained(
            model,
            init_adapter.as_posix(),
            is_trainable=True,
        )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    output_dir = Path(str(cfg.student.output_dir))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        optim=str(cfg.student.training.optim),
        num_train_epochs=float(cfg.student.training.num_train_epochs),
        max_steps=int(cfg.student.training.max_steps),
        learning_rate=float(cfg.student.training.learning_rate),
        warmup_steps=training_warmup_steps(cfg.student.training),
        weight_decay=float(cfg.student.training.weight_decay),
        per_device_train_batch_size=int(cfg.student.training.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.student.training.gradient_accumulation_steps),
        logging_steps=int(cfg.student.training.logging_steps),
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    optimizer_name = str(cfg.student.training.optimizer)
    if optimizer_name not in {"adamw", "muon"}:
        raise ValueError(f"unknown student.training.optimizer={optimizer_name!r}")
    trainer_cls = MuonSFTTrainer if optimizer_name == "muon" else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir.parent / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
