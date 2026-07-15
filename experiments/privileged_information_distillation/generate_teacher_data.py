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
    append_reference_material,
    build_student_prompt,
    build_teacher_prompt,
    extract_harmony_final,
    format_student_target,
    parse_counterfactual_teacher_target,
    parse_teacher_target,
    route_reference_material,
)
from experiments.privileged_information_distillation.evaluate_student_sft import (
    load_retrieval_cache,
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
    dataset_name_contains = OmegaConf.select(
        cfg, "teacher.dataset_name_contains", default=None
    )
    retrieval_cache = OmegaConf.select(cfg, "teacher.retrieval_cache", default=None)
    references: dict[tuple[str, Any], str] = {}
    if retrieval_cache is not None:
        retrieval_path = Path(str(retrieval_cache))
        if not retrieval_path.is_absolute():
            retrieval_path = root / retrieval_path
        passage_field = str(OmegaConf.select(
            cfg, "teacher.retrieval_passage_field", default="passages"
        ))
        references = load_retrieval_cache(retrieval_path, passage_field=passage_field)
    reference_visibility = str(OmegaConf.select(
        cfg, "teacher.reference_visibility", default="teacher_and_student"
    ))
    rows: list[dict[str, Any]] = []
    for dataset_cfg in datasets:
        if (
            dataset_name_contains is not None
            and str(dataset_name_contains) not in dataset_cfg.name
        ):
            continue
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
                include_reasoning=bool(OmegaConf.select(
                    cfg, "student.include_reasoning", default=False
                )),
                reasoning_max_chars=int(OmegaConf.select(
                    cfg, "student.reasoning_max_chars", default=0
                )),
                reasoning_truncation=str(OmegaConf.select(
                    cfg, "student.reasoning_truncation", default="head_tail"
                )),
            )
            if retrieval_cache is not None:
                key = (dataset_cfg.name, index)
                if key not in references:
                    raise RuntimeError(
                        f"retrieval cache is missing dataset={dataset_cfg.name!r} "
                        f"index={index!r}"
                    )
                student_prompt, teacher_input_prompt = route_reference_material(
                    student_prompt,
                    references[key],
                    reference_visibility,
                )
            else:
                teacher_input_prompt = student_prompt
            rows.append({
                "dataset": dataset_cfg.name,
                "index": index,
                "label": label,
                "student_prompt": student_prompt,
                "reference_visibility": (
                    reference_visibility if retrieval_cache is not None else None
                ),
                "teacher_prompt": build_teacher_prompt(
                    teacher_input_prompt,
                    str(cfg.teacher.prompt),
                    label,
                ),
            })
    if dataset_name_contains is not None:
        rows = filter_teacher_rows_by_dataset(rows, str(dataset_name_contains))
    return limit_teacher_rows(
        rows,
        limit=cfg.teacher.limit,
        limit_per_label=OmegaConf.select(cfg, "teacher.limit_per_label", default=None),
    )


def filter_teacher_rows_by_dataset(
    rows: list[dict[str, Any]], dataset_name_contains: str
) -> list[dict[str, Any]]:
    """Restrict teacher generation to datasets used by a specialized student."""
    selected = [
        row for row in rows
        if dataset_name_contains in str(row.get("dataset", ""))
    ]
    if not selected:
        raise RuntimeError(
            f"no teacher rows match dataset_name_contains={dataset_name_contains!r}"
        )
    return selected


def limit_teacher_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    limit_per_label: int | None,
) -> list[dict[str, Any]]:
    """Apply either a simple limit or a balanced per-label smoke limit."""
    if limit is not None and limit_per_label is not None:
        raise ValueError("set only one of teacher.limit and teacher.limit_per_label")
    if limit_per_label is None:
        return rows if limit is None else rows[:int(limit)]
    selected: list[dict[str, Any]] = []
    counts = {0: 0, 1: 0}
    wanted = int(limit_per_label)
    for row in rows:
        label = int(row["label"])
        if label in counts and counts[label] < wanted:
            selected.append(row)
            counts[label] += 1
        if all(count == wanted for count in counts.values()):
            break
    if any(count < wanted for count in counts.values()):
        raise RuntimeError(f"could not select {wanted} teacher rows per label: {counts}")
    return selected


def load_cached_records(path: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, Any], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[(record["dataset"], record["index"])] = record
    return records


def cache_matches(row: dict[str, Any], cached: dict[str, Any] | None) -> bool:
    if not cached:
        return False
    return (
        cached.get("label") == row["label"]
        and cached.get("student_prompt") == row["student_prompt"]
        and cached.get("teacher_prompt") == row["teacher_prompt"]
        and not cached.get("parse_error", True)
        and cached.get("label_match") is True
        and cached.get("student_target")
    )


def reparse_cached_record(
    row: dict[str, Any],
    cached: dict[str, Any] | None,
    target_format: str = "summary",
) -> dict[str, Any] | None:
    if not cached or not cached.get("raw_completion"):
        return cached
    if (
        cached.get("label") != row["label"]
        or cached.get("student_prompt") != row["student_prompt"]
        or cached.get("teacher_prompt") != row["teacher_prompt"]
    ):
        return cached
    parser = (
        parse_counterfactual_teacher_target
        if target_format == "counterfactual"
        else parse_teacher_target
    )
    parsed = parser(cached["raw_completion"], expected_prediction=row["label"])
    if not parsed:
        return cached
    if target_format == "counterfactual":
        summary, facts, contradiction, prediction = parsed
    else:
        summary, prediction = parsed
        facts = contradiction = None
    return {
        **cached,
        "reasoning_summary": summary,
        "facts": facts,
        "contradiction": contradiction,
        "prediction": prediction,
        "student_target": format_student_target(
            summary, prediction, facts=facts, contradiction=contradiction
        ),
        "parse_error": False,
        "label_match": prediction == row["label"],
        "prediction_source": (
            "teacher_final" if "Prediction:" in cached.get("harmony_final", "")
            else "privileged_label_fallback"
        ),
    }


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
    target_format = str(OmegaConf.select(cfg, "student.target_format", default="summary"))
    if target_format not in {"summary", "counterfactual"}:
        raise ValueError(f"unknown student.target_format={target_format!r}")
    print(f"loaded {len(rows)} privileged teacher examples")

    artifact = Path(str(cfg.teacher.artifact))
    if not artifact.is_absolute():
        artifact = root / artifact
    cached = (
        {} if bool(cfg.teacher.force_regenerate)
        else load_cached_records(artifact)
    )
    reusable: dict[tuple[str, Any], dict[str, Any]] = {}
    missing_rows = []
    for row in rows:
        key = (row["dataset"], row["index"])
        refreshed = reparse_cached_record(row, cached.get(key), target_format)
        if cache_matches(row, refreshed):
            reusable[key] = refreshed
        else:
            missing_rows.append(row)
    print(f"cache hits={len(reusable)} generation required={len(missing_rows)}")

    generated: dict[tuple[str, Any], str] = {}
    if missing_rows:
        tokenizer = AutoTokenizer.from_pretrained(str(cfg.teacher.model))
        prompts = [render_chat_prompt(tokenizer, row["teacher_prompt"]) for row in missing_rows]
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
        generated = {
            (row["dataset"], row["index"]): (
                output.outputs[0].text if output.outputs else ""
            )
            for row, output in zip(missing_rows, outputs, strict=True)
        }

    artifact.parent.mkdir(parents=True, exist_ok=True)
    parsed_count = 0
    label_match_count = 0
    temporary = artifact.with_suffix(artifact.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            key = (row["dataset"], row["index"])
            if key in reusable:
                record = reusable[key]
                parsed_count += 1
                label_match_count += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            raw_completion = generated[key]
            parser = (
                parse_counterfactual_teacher_target
                if target_format == "counterfactual"
                else parse_teacher_target
            )
            parsed = parser(raw_completion, expected_prediction=row["label"])
            if parsed and target_format == "counterfactual":
                summary, facts, contradiction, prediction = parsed
            elif parsed:
                summary, prediction = parsed
                facts = contradiction = None
            else:
                summary = facts = contradiction = prediction = None
            parsed_count += int(parsed is not None)
            label_match_count += int(prediction == row["label"])
            record = {
                **row,
                "reasoning_summary": summary,
                "facts": facts,
                "contradiction": contradiction,
                "prediction": prediction,
                "student_target": (
                    format_student_target(
                        summary, prediction, facts=facts, contradiction=contradiction
                    ) if parsed else None
                ),
                "parse_error": parsed is None,
                "label_match": prediction == row["label"],
                "prediction_source": (
                    "teacher_final" if "Prediction:" in extract_harmony_final(raw_completion)
                    else "privileged_label_fallback"
                ),
                "harmony_final": extract_harmony_final(raw_completion),
                "raw_completion": raw_completion,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(artifact)

    print(f"parsed={parsed_count}/{len(rows)} label_matches={label_match_count}/{len(rows)}")
    print(f"wrote {artifact}")
    (artifact.parent / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
