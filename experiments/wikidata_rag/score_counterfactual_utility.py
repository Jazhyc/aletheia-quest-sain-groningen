#!/usr/bin/env python3
"""Measure each retrieved fact's effect on a frozen student's binary margin."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    append_reference_material,
    build_student_prompt,
)
from experiments.privileged_information_distillation.evaluate_student_sft import (
    EMPTY_REASONING_PREFIX,
    binary_score_from_logprobs,
    binary_token_ids,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import (
    load_labels,
    load_split_config,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def format_candidate_reference(candidate: dict[str, Any]) -> str:
    """Render one compact candidate with the same syntax used by judge sweeps."""
    return f"- {candidate.get('subject', '')}: {candidate.get('fact', '')}"


def correct_log_probability(score: float, label: int, epsilon: float = 1e-7) -> float:
    probability = score if label else 1.0 - score
    return math.log(max(epsilon, min(1.0, probability)))


def attach_semantic_labels(
    rows: list[dict[str, Any]], supervision_path: Path | None
) -> None:
    if supervision_path is None:
        return
    supervised = {
        (row["dataset"], row["index"]): row
        for row in load_jsonl(supervision_path)
        if not row.get("parse_error")
    }
    for row in rows:
        source = supervised.get((row["dataset"], row["index"]))
        labels = {
            item["id"]: item["label"] for item in (source or {}).get("labels", [])
        }
        for candidate in row["candidates"]:
            if candidate["id"] in labels:
                candidate["semantic_label"] = labels[candidate["id"]]


def prompt_lookup(
    rows: list[dict[str, Any]],
    split: str,
    splits_dir: Path,
    config: dict[str, Any],
    tokenizer: Any | None = None,
) -> tuple[dict[tuple[str, Any], tuple[str, int]], Any]:
    """Load only the public rows requested by the bounded candidate cache."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config["student"]["output_dir"])
    wanted: dict[str, set[Any]] = {}
    for row in rows:
        wanted.setdefault(row["dataset"], set()).add(row["index"])
    found: dict[tuple[str, Any], tuple[str, int]] = {}
    split_configs = {
        item.name: item for item in load_split_config(splits_dir / f"dry.{split}.yaml", ROOT)
    }
    for dataset_name, indices in wanted.items():
        if dataset_name not in split_configs:
            raise KeyError(f"{dataset_name} is absent from dry.{split}.yaml")
        dataset_cfg = split_configs[dataset_name]
        labels = load_labels(dataset_cfg)
        label_by_index = dict(zip(labels["index"], labels["label"], strict=True))
        dataset = load_dataset(dataset_name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        for record in dataset:
            index = record["index"]
            if index not in indices:
                continue
            raw = build_student_prompt(
                record["messages"],
                config["student"]["prompt"],
                int(config["student"]["max_prompt_chars"]),
                config["student"]["context_truncation"],
            )
            found[(dataset_name, index)] = (raw, int(label_by_index[index]))
    missing = {
        (row["dataset"], row["index"]) for row in rows
    } - set(found)
    if missing:
        raise RuntimeError(f"failed to load {len(missing)} requested rows")
    return found, tokenizer


def rotated_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Give each row one deterministic candidate from a different question group."""
    donors = [next(iter(row["candidates"]), None) for row in rows]
    if len(rows) < 2:
        return [None] * len(rows)
    controls: list[dict[str, Any] | None] = []
    for index, row in enumerate(rows):
        donor = None
        for offset in range(1, len(rows) + 1):
            other_index = (index + offset) % len(rows)
            if (
                rows[other_index]["question_group"] != row["question_group"]
                and donors[other_index] is not None
            ):
                donor = dict(donors[other_index])
                donor["donor_dataset"] = rows[other_index]["dataset"]
                donor["donor_index"] = rows[other_index]["index"]
                break
        controls.append(donor)
    return controls


def extract_binary_scores(outputs: list[Any], token_ids: list[int]) -> list[float]:
    scores: list[float] = []
    for output in outputs:
        logprobs = {}
        if output.outputs and output.outputs[0].logprobs:
            logprobs = output.outputs[0].logprobs[0] or {}
        score = binary_score_from_logprobs(logprobs, token_ids)
        if score is None:
            raise RuntimeError("vLLM omitted a requested binary token log probability")
        scores.append(score)
    return scores


def score_rows(
    llm: Any,
    sampling: Any,
    lora_request: Any,
    rows: list[dict[str, Any]],
    prompts: dict[tuple[str, Any], tuple[str, int]],
    tokenizer: Any,
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    token_ids = binary_token_ids(tokenizer)
    controls = rotated_controls(rows)
    requests: list[tuple[int, str, int | None, str]] = []
    rendered: list[str] = []
    for row_number, (row, control) in enumerate(zip(rows, controls, strict=True)):
        raw_prompt, _ = prompts[(row["dataset"], row["index"])]
        references: list[tuple[str, int | None, str]] = [("empty", None, "")]
        references.extend(
            ("candidate", candidate_number, format_candidate_reference(candidate))
            for candidate_number, candidate in enumerate(row["candidates"])
        )
        if control is not None:
            references.append(("shuffled", None, format_candidate_reference(control)))
        for kind, candidate_number, reference in references:
            augmented = append_reference_material(raw_prompt, reference)
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": augmented}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rendered.append(chat + EMPTY_REASONING_PREFIX)
            requests.append((row_number, kind, candidate_number, reference))

    scores: list[float] = []
    started = time.time()
    for start in range(0, len(rendered), batch_size):
        stop = min(len(rendered), start + batch_size)
        outputs = llm.generate(
            rendered[start:stop], sampling, lora_request=lora_request, use_tqdm=False
        )
        scores.extend(extract_binary_scores(outputs, token_ids))
        print(f"scored {stop}/{len(rendered)} prefixes", flush=True)
    elapsed = time.time() - started

    result = [dict(row) for row in rows]
    for row in result:
        _, label = prompts[(row["dataset"], row["index"])]
        row["label"] = label
    for (row_number, kind, candidate_number, _), score in zip(requests, scores, strict=True):
        row = result[row_number]
        label = row["label"]
        logp = correct_log_probability(score, label)
        if kind == "empty":
            row["empty_score"] = score
            row["empty_correct_logprob"] = logp
        elif kind == "shuffled":
            control = controls[row_number]
            row["shuffled_control"] = {
                **(control or {}), "score": score, "correct_logprob": logp,
            }
        else:
            candidate = row["candidates"][int(candidate_number)]
            candidate["score"] = score
            candidate["correct_logprob"] = logp

    for row in result:
        empty_logp = row["empty_correct_logprob"]
        shuffled_logp = row.get("shuffled_control", {}).get(
            "correct_logprob", empty_logp
        )
        shuffled_utility = shuffled_logp - empty_logp
        row["shuffled_utility"] = shuffled_utility
        for candidate in row["candidates"]:
            raw_utility = candidate["correct_logprob"] - empty_logp
            candidate["utility"] = raw_utility
            candidate["controlled_utility"] = raw_utility - shuffled_utility
    timing = {
        "rows": len(rows),
        "candidate_prefixes": sum(len(row["candidates"]) for row in rows),
        "total_prefixes": len(rendered),
        "score_seconds": elapsed,
        "prefixes_per_second": len(rendered) / elapsed,
    }
    return result, timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--supervision-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--validation-input", type=Path)
    parser.add_argument("--validation-supervision-input", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.resolve()
    config = yaml.safe_load((adapter_dir.parent / "config.yaml").read_text())
    rows = load_jsonl(args.input.resolve())
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    attach_semantic_labels(
        rows,
        args.supervision_input.resolve() if args.supervision_input is not None else None,
    )
    prompt_by_key, tokenizer = prompt_lookup(
        rows, args.split, args.splits_dir.resolve(), config
    )
    validation_rows: list[dict[str, Any]] = []
    validation_prompts: dict[tuple[str, Any], tuple[str, int]] = {}
    validation_arguments = (
        args.validation_input, args.validation_output, args.validation_report
    )
    if any(value is not None for value in validation_arguments):
        if not all(value is not None for value in validation_arguments):
            raise SystemExit(
                "--validation-input, --validation-output, and --validation-report "
                "must be supplied together"
            )
        validation_rows = load_jsonl(args.validation_input.resolve())
        if args.limit_rows is not None:
            validation_rows = validation_rows[: args.limit_rows]
        attach_semantic_labels(
            validation_rows,
            args.validation_supervision_input.resolve()
            if args.validation_supervision_input is not None else None,
        )
        validation_prompts, _ = prompt_lookup(
            validation_rows, "validation", args.splits_dir.resolve(), config, tokenizer
        )

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=config["student"]["model"],
        tokenizer=adapter_dir.as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        enable_prefix_caching=True,
        max_lora_rank=int(config["student"]["lora"]["r"]),
        max_model_len=args.max_model_len,
    )
    binary_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(binary_ids),
        logprob_token_ids=binary_ids,
        allowed_token_ids=binary_ids,
    )
    request = LoRARequest(adapter_dir.parent.name, 1, adapter_dir.as_posix())
    scored, timing = score_rows(
        llm, sampling, request, rows, prompt_by_key, tokenizer,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in scored) + "\n"
    )
    args.report.write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2), flush=True)
    if validation_rows:
        scored, timing = score_rows(
            llm, sampling, request, validation_rows, validation_prompts, tokenizer,
            batch_size=args.batch_size,
        )
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in scored) + "\n"
        )
        args.validation_report.write_text(json.dumps(timing, indent=2) + "\n")
        print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
