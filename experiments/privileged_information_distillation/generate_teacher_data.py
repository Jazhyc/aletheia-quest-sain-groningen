#!/usr/bin/env python3
"""Generate privileged GPT-OSS reasoning summaries for Qwen distillation."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    build_student_prompt,
    build_teacher_prompt,
    extract_harmony_final,
    format_student_target,
    parse_teacher_target,
)
from experiments.qwen_grpo_lora.run_qwen_grpo_lora import (
    load_labels,
    load_split_config,
)


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_teacher_rows(cfg: DictConfig, root: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    split_path = root / str(cfg.splits_dir) / f"dry.{cfg.teacher.split}.yaml"
    datasets = load_split_config(split_path, root)
    rows: list[dict[str, Any]] = []
    for dataset_cfg in datasets:
        labels = load_labels(dataset_cfg)
        label_by_index = dict(zip(labels["index"], labels["label"], strict=True))
        wanted = set(label_by_index)
        dataset = load_dataset(dataset_cfg.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        for example in dataset:
            index = example["index"]
            if index not in wanted:
                continue
            label = int(label_by_index[index])
            student_prompt = build_student_prompt(
                example["messages"],
                str(cfg.student.prompt),
                int(cfg.student.max_prompt_chars),
                str(cfg.student.context_truncation),
            )
            rows.append({
                "dataset": dataset_cfg.name,
                "index": index,
                "label": label,
                "student_prompt": student_prompt,
                "teacher_prompt": build_teacher_prompt(
                    student_prompt,
                    str(cfg.teacher.prompt),
                    label,
                ),
            })
    limit = cfg.teacher.limit
    return rows if limit is None else rows[:int(limit)]


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="privileged_information_distillation",
)
def main(cfg: DictConfig) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    root = Path(get_original_cwd()).resolve()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    rows = load_teacher_rows(cfg, root)
    print(f"loaded {len(rows)} privileged teacher examples")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.teacher.model))
    prompts = [render_chat_prompt(tokenizer, row["teacher_prompt"]) for row in rows]
    llm = LLM(
        model=str(cfg.teacher.model),
        dtype=str(cfg.teacher.dtype),
        max_model_len=int(cfg.teacher.max_model_len),
        gpu_memory_utilization=float(cfg.teacher.gpu_memory_utilization),
        seed=int(cfg.seed),
    )
    sampling = SamplingParams(
        max_tokens=int(cfg.teacher.max_tokens),
        temperature=float(cfg.teacher.temperature),
    )
    batch_size = cfg.teacher.batch_size
    outputs = []
    if batch_size is None:
        outputs = list(llm.generate(prompts, sampling))
    else:
        for start in range(0, len(prompts), int(batch_size)):
            outputs.extend(llm.generate(prompts[start:start + int(batch_size)], sampling))

    artifact = Path(str(cfg.teacher.artifact))
    if not artifact.is_absolute():
        artifact = root / artifact
    artifact.parent.mkdir(parents=True, exist_ok=True)
    parsed_count = 0
    label_match_count = 0
    with artifact.open("w") as handle:
        for row, output in zip(rows, outputs, strict=True):
            raw_completion = output.outputs[0].text if output.outputs else ""
            parsed = parse_teacher_target(raw_completion)
            summary = parsed[0] if parsed else None
            prediction = parsed[1] if parsed else None
            parsed_count += int(parsed is not None)
            label_match_count += int(prediction == row["label"])
            record = {
                **row,
                "reasoning_summary": summary,
                "prediction": prediction,
                "student_target": (
                    format_student_target(summary, prediction) if parsed else None
                ),
                "parse_error": parsed is None,
                "label_match": prediction == row["label"],
                "harmony_final": extract_harmony_final(raw_completion),
                "raw_completion": raw_completion,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"parsed={parsed_count}/{len(rows)} label_matches={label_match_count}/{len(rows)}")
    print(f"wrote {artifact}")
    (artifact.parent / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
