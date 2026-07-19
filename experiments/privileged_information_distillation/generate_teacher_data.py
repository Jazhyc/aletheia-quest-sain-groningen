#!/usr/bin/env python3
"""Generate privileged GPT-OSS reasoning summaries for Qwen distillation."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
    extract_teacher_final,
    format_student_target,
    parse_counterfactual_teacher_target,
    parse_rating_teacher_target,
    parse_teacher_target,
    rating_matches_prediction,
    route_reference_material,
    split_qwen_think_completion,
)
from experiments.privileged_information_distillation.evaluate_student_sft import (
    load_retrieval_cache,
)
from experiments.qwen_grpo_lora.run_qwen_grpo_lora import (
    load_labels,
    load_split_config,
)


REASONING_EFFORTS = {"low", "medium", "high"}
TEACHER_OUTPUT_FORMATS = {"harmony", "qwen_think"}


def teacher_expected_prediction(row: dict[str, Any]) -> int | None:
    """Return the privileged fallback only when the teacher is allowed the label."""
    if not bool(row.get("teacher_uses_ground_truth", True)):
        return None
    return int(row["label"])


def build_configured_teacher_prompt(
    student_prompt: str,
    teacher_template: str,
    label: int,
    *,
    uses_ground_truth: bool,
) -> str:
    """Render a teacher prompt while enforcing blind-mode label isolation."""
    if not uses_ground_truth and "__GROUND_TRUTH__" in teacher_template:
        raise ValueError(
            "blind teacher prompt must not contain __GROUND_TRUTH__"
        )
    return build_teacher_prompt(student_prompt, teacher_template, label)


def normalize_reasoning_effort(value: Any) -> str:
    effort = str(value).lower()
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            f"teacher.reasoning_effort must be one of {sorted(REASONING_EFFORTS)}, "
            f"got {value!r}"
        )
    return effort


def normalize_teacher_output_format(value: Any) -> str:
    output_format = str(value).lower()
    if output_format not in TEACHER_OUTPUT_FORMATS:
        raise ValueError(
            "teacher.output_format must be one of "
            f"{sorted(TEACHER_OUTPUT_FORMATS)}, got {value!r}"
        )
    return output_format


def render_chat_prompt(
    tokenizer: Any,
    prompt: str,
    reasoning_effort: str,
    output_format: str = "harmony",
) -> str:
    output_format = normalize_teacher_output_format(output_format)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if output_format == "harmony":
        kwargs["reasoning_effort"] = normalize_reasoning_effort(reasoning_effort)
    else:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **kwargs,
    )


def generate_openai_completions(
    prompts: list[str],
    *,
    api_base: str,
    api_key: str,
    served_model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    request_timeout: float,
) -> list[str]:
    """Generate raw completions through one persistent OpenAI-compatible server."""
    import requests

    endpoint = api_base.rstrip("/") + "/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def generate(prompt: str) -> str:
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": served_model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0].get("text", ""))

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        return list(executor.map(generate, prompts))


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
    reasoning_effort = normalize_reasoning_effort(
        OmegaConf.select(cfg, "teacher.reasoning_effort", default="medium")
    )
    teacher_output_format = normalize_teacher_output_format(
        OmegaConf.select(cfg, "teacher.output_format", default="harmony")
    )
    teacher_uses_ground_truth = bool(OmegaConf.select(
        cfg, "teacher.uses_ground_truth", default=True
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
                "teacher_model": str(cfg.teacher.model),
                "teacher_output_format": teacher_output_format,
                "reasoning_effort": reasoning_effort,
                "teacher_uses_ground_truth": teacher_uses_ground_truth,
                "student_prompt": student_prompt,
                "reference_visibility": (
                    reference_visibility if retrieval_cache is not None else None
                ),
                "teacher_prompt": build_configured_teacher_prompt(
                    teacher_input_prompt,
                    str(cfg.teacher.prompt),
                    label,
                    uses_ground_truth=teacher_uses_ground_truth,
                ),
            })
    if dataset_name_contains is not None:
        rows = filter_teacher_rows_by_dataset(rows, str(dataset_name_contains))
    selection_manifest = OmegaConf.select(
        cfg, "teacher.selection_manifest", default=None
    )
    if selection_manifest is not None:
        selection_path = Path(str(selection_manifest))
        if not selection_path.is_absolute():
            selection_path = root / selection_path
        rows = select_teacher_rows_by_manifest(rows, selection_path)
    rows = shard_teacher_rows(
        rows,
        shard_count=int(OmegaConf.select(cfg, "teacher.shard_count", default=1)),
        shard_index=int(OmegaConf.select(cfg, "teacher.shard_index", default=0)),
    )
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


def select_teacher_rows_by_manifest(
    rows: list[dict[str, Any]], manifest_path: Path
) -> list[dict[str, Any]]:
    """Select exact generation rows without exposing their labels in blind prompts."""
    desired = {
        (str(record["dataset"]), str(record["index"])): int(record["label"])
        for record in (
            json.loads(line)
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        )
    }
    selected = [
        row for row in rows
        if (str(row["dataset"]), str(row["index"])) in desired
    ]
    available = {
        (str(row["dataset"]), str(row["index"])): int(row["label"])
        for row in selected
    }
    if set(available) != set(desired):
        missing = sorted(set(desired) - set(available))
        raise ValueError(
            f"teacher selection manifest has unavailable rows; first={missing[0]}"
        )
    mismatched = [key for key in desired if desired[key] != available[key]]
    if mismatched:
        raise ValueError(f"teacher selection label mismatch for {mismatched[0]}")
    return selected


def shard_teacher_rows(
    rows: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    """Deterministically shard every dataset/label stratum round-robin."""
    if shard_count < 1:
        raise ValueError(f"teacher.shard_count must be positive, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            "teacher.shard_index must satisfy "
            f"0 <= shard_index < shard_count, got {shard_index}/{shard_count}"
        )
    if shard_count == 1:
        return rows
    offsets: defaultdict[tuple[str, int], int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for row in rows:
        stratum = (str(row["dataset"]), int(row["label"]))
        offset = offsets[stratum]
        offsets[stratum] += 1
        if offset % shard_count == shard_index:
            selected.append(row)
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
        and cached.get("teacher_model", "openai/gpt-oss-120b")
        == row.get("teacher_model", "openai/gpt-oss-120b")
        and cached.get("teacher_output_format", "harmony")
        == row.get("teacher_output_format", "harmony")
        and cached.get("reasoning_effort", "medium") == row["reasoning_effort"]
        and cached.get("teacher_uses_ground_truth", True)
        == row.get("teacher_uses_ground_truth", True)
        and cached.get("student_prompt") == row["student_prompt"]
        and cached.get("teacher_prompt") == row["teacher_prompt"]
        and not cached.get("parse_error", True)
        and (
            not row.get("teacher_uses_ground_truth", True)
            or cached.get("label_match") is True
        )
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
        or cached.get("teacher_model", "openai/gpt-oss-120b")
        != row.get("teacher_model", "openai/gpt-oss-120b")
        or cached.get("teacher_output_format", "harmony")
        != row.get("teacher_output_format", "harmony")
        or cached.get("reasoning_effort", "medium") != row["reasoning_effort"]
        or cached.get("teacher_uses_ground_truth", True)
        != row.get("teacher_uses_ground_truth", True)
        or cached.get("student_prompt") != row["student_prompt"]
        or cached.get("teacher_prompt") != row["teacher_prompt"]
    ):
        return cached
    parser = {
        "summary": parse_teacher_target,
        "counterfactual": parse_counterfactual_teacher_target,
        "summary_rating": parse_rating_teacher_target,
    }[target_format]
    parsed = parser(
        cached["raw_completion"],
        expected_prediction=teacher_expected_prediction(row),
        output_format=row.get("teacher_output_format", "harmony"),
    )
    if not parsed:
        return cached
    if target_format == "counterfactual":
        summary, facts, contradiction, prediction = parsed
        rating = None
    elif target_format == "summary_rating":
        summary, rating, prediction = parsed
        facts = contradiction = None
    else:
        summary, prediction = parsed
        facts = contradiction = None
        rating = None
    rating_polarity_match = (
        rating is None or rating_matches_prediction(rating, prediction)
    )
    return {
        **cached,
        "reasoning_summary": summary,
        "facts": facts,
        "contradiction": contradiction,
        "rating": rating,
        "rating_polarity_match": rating_polarity_match,
        "prediction": prediction,
        "student_target": format_student_target(
            summary,
            prediction,
            facts=facts,
            contradiction=contradiction,
            rating=rating,
        ),
        "parse_error": not rating_polarity_match,
        "label_match": prediction == row["label"],
        "prediction_source": (
            "teacher_final" if "Prediction:" in (
                extract_teacher_final(
                    cached["raw_completion"],
                    row.get("teacher_output_format", "harmony"),
                ) or ""
            )
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

    root = Path(get_original_cwd()).resolve()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    rows = load_teacher_rows(cfg, root)
    target_format = str(OmegaConf.select(cfg, "student.target_format", default="summary"))
    if target_format not in {"summary", "counterfactual", "summary_rating"}:
        raise ValueError(f"unknown student.target_format={target_format!r}")
    teacher_mode = (
        "privileged" if all(
            row.get("teacher_uses_ground_truth", True) for row in rows
        ) else "blind"
    )
    print(f"loaded {len(rows)} {teacher_mode} teacher examples")

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
        prompts = [
            render_chat_prompt(
                tokenizer,
                row["teacher_prompt"],
                row["reasoning_effort"],
                row["teacher_output_format"],
            )
            for row in missing_rows
        ]
        backend = str(OmegaConf.select(cfg, "teacher.backend", default="offline"))
        if backend == "openai":
            completion_texts = generate_openai_completions(
                prompts,
                api_base=str(cfg.teacher.api_base),
                api_key=str(cfg.teacher.api_key),
                served_model=str(cfg.teacher.served_model),
                max_tokens=int(cfg.teacher.max_tokens),
                temperature=float(cfg.teacher.temperature),
                concurrency=int(cfg.teacher.api_concurrency),
                request_timeout=float(cfg.teacher.request_timeout),
            )
        elif backend == "offline":
            from vllm import LLM, SamplingParams

            llm_kwargs: dict[str, Any] = dict(
                model=str(cfg.teacher.model),
                dtype=str(cfg.teacher.dtype),
                max_model_len=int(cfg.teacher.max_model_len),
                gpu_memory_utilization=float(cfg.teacher.gpu_memory_utilization),
                seed=int(cfg.seed),
            )
            for key in ("tensor_parallel_size", "max_num_seqs"):
                value = OmegaConf.select(cfg, f"teacher.{key}", default=None)
                if value is not None:
                    llm_kwargs[key] = int(value)
            enable_prefix_caching = OmegaConf.select(
                cfg, "teacher.enable_prefix_caching", default=None
            )
            if enable_prefix_caching is not None:
                llm_kwargs["enable_prefix_caching"] = bool(enable_prefix_caching)
            llm = LLM(**llm_kwargs)
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
            completion_texts = [
                output.outputs[0].text if output.outputs else ""
                for output in outputs
            ]
        else:
            raise ValueError(f"unknown teacher.backend={backend!r}")
        generated = {
            (row["dataset"], row["index"]): text
            for row, text in zip(missing_rows, completion_texts, strict=True)
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
                label_match_count += int(record.get("label_match") is True)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            raw_completion = generated[key]
            parser = {
                "summary": parse_teacher_target,
                "counterfactual": parse_counterfactual_teacher_target,
                "summary_rating": parse_rating_teacher_target,
            }[target_format]
            output_format = row.get("teacher_output_format", "harmony")
            parsed = parser(
                raw_completion,
                expected_prediction=teacher_expected_prediction(row),
                output_format=output_format,
            )
            if parsed and target_format == "counterfactual":
                summary, facts, contradiction, prediction = parsed
                rating = None
            elif parsed and target_format == "summary_rating":
                summary, rating, prediction = parsed
                facts = contradiction = None
            elif parsed:
                summary, prediction = parsed
                facts = contradiction = None
                rating = None
            else:
                summary = facts = contradiction = rating = prediction = None
            parsed_count += int(parsed is not None)
            label_match_count += int(prediction == row["label"])
            rating_polarity_match = (
                parsed is not None
                and (
                    rating is None
                    or rating_matches_prediction(rating, prediction)
                )
            )
            teacher_final = extract_teacher_final(raw_completion, output_format)
            qwen_split = (
                split_qwen_think_completion(raw_completion)
                if output_format == "qwen_think"
                else None
            )
            record = {
                **row,
                "reasoning_summary": summary,
                "facts": facts,
                "contradiction": contradiction,
                "rating": rating,
                "rating_polarity_match": rating_polarity_match,
                "prediction": prediction,
                "student_target": (
                    format_student_target(
                        summary,
                        prediction,
                        facts=facts,
                        contradiction=contradiction,
                        rating=rating,
                    ) if parsed else None
                ),
                "parse_error": parsed is None or not rating_polarity_match,
                "label_match": prediction == row["label"],
                "prediction_source": (
                    "teacher_final" if "Prediction:" in (teacher_final or "")
                    else "privileged_label_fallback"
                ),
                # Keep the legacy field for existing audit scripts while making
                # the model-agnostic visible-final provenance explicit.
                "harmony_final": teacher_final or "",
                "teacher_final": teacher_final or "",
                "private_reasoning_chars": len(qwen_split[0]) if qwen_split else None,
                "raw_completion": raw_completion,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(artifact)

    print(f"parsed={parsed_count}/{len(rows)} label_matches={label_match_count}/{len(rows)}")
    print(f"wrote {artifact}")
    (artifact.parent / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
