#!/usr/bin/env python3
"""SFT a Qwen LoRA on concise privileged-teacher reasoning summaries."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


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


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    usable = [
        record for record in records
        if not record.get("parse_error")
        and record.get("label_match")
        and record.get("student_target")
    ]
    if not usable:
        raise RuntimeError(f"no usable teacher records in {path}")
    return usable


def tokenize_record(record: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": record["student_prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(
        record["student_target"] + (tokenizer.eos_token or ""),
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
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    root = Path(get_original_cwd()).resolve()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    artifact = Path(str(cfg.teacher.artifact))
    if not artifact.is_absolute():
        artifact = root / artifact
    records = load_records(artifact)
    if cfg.student.train_limit is not None:
        records = records[:int(cfg.student.train_limit)]
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.student.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = [
        tokenize_record(record, tokenizer, int(cfg.student.max_length))
        for record in records
    ]
    dataset = Dataset.from_list(tokenized)
    print(f"training on {len(dataset)} parsed, label-consistent teacher targets")

    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.student.model),
        torch_dtype=torch.bfloat16,
    )
    model = get_peft_model(model, LoraConfig(
        r=int(cfg.student.lora.r),
        lora_alpha=int(cfg.student.lora.alpha),
        lora_dropout=float(cfg.student.lora.dropout),
        target_modules=list(cfg.student.lora.target_modules),
        task_type="CAUSAL_LM",
    ))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    output_dir = Path(str(cfg.student.output_dir))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        num_train_epochs=float(cfg.student.training.num_train_epochs),
        max_steps=int(cfg.student.training.max_steps),
        learning_rate=float(cfg.student.training.learning_rate),
        warmup_ratio=float(cfg.student.training.warmup_ratio),
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
    trainer = Trainer(
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
