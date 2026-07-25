#!/usr/bin/env python3
"""Evaluate one frozen semantic prompt route without generating unused prompts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.compose_prompt_router import route_matches
from experiments.liars_bench_distillation.evaluate_prompt_sweep import (
    load_prompt,
    split_records,
    summarize,
)
from experiments.liars_bench_distillation.evaluate_students import parse_prediction
from experiments.privileged_information_distillation.core import build_student_prompt


def render(
    tokenizer: Any,
    records: list[dict[str, Any]],
    prompt_template: str,
    *,
    max_prompt_chars: int,
    context_truncation: str,
) -> list[str]:
    """Render the same no-thinking chat contract for a list of records."""
    return [
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": build_student_prompt(
                    row["messages"],
                    prompt_template,
                    max_prompt_chars,
                    context_truncation,
                ),
            }],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in records
    ]


def evaluated_rows(
    records: list[dict[str, Any]],
    prompts: list[str],
    outputs: list[Any],
) -> list[dict[str, Any]]:
    """Convert vLLM outputs to the shared external-generation schema."""
    rows = []
    for record, prompt, output in zip(records, prompts, outputs, strict=True):
        generation = output.outputs[0].text if output.outputs else ""
        parsed = parse_prediction(generation)
        rows.append({
            **{
                key: record[key]
                for key in (
                    "dataset",
                    "index",
                    "category",
                    "source_model",
                    "label",
                )
            },
            "prediction": 0 if parsed is None else parsed,
            "parse_error": parsed is None,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "generation": generation,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--specialist-config", type=Path, required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument(
        "--route-kind",
        choices=("knowledge", "choice", "union"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "confirmation", "all"),
        required=True,
    )
    parser.add_argument("--split-seed", type=int, default=20260725)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument(
        "--context-truncation",
        choices=("head_tail", "tail"),
        default="head_tail",
    )
    args = parser.parse_args()

    all_records = [
        json.loads(line)
        for line in args.eval_artifact.read_text().splitlines()
        if line.strip()
    ]
    records = split_records(all_records, args.split, seed=args.split_seed)
    if args.expected_rows is not None and len(records) != args.expected_rows:
        raise ValueError(
            f"{args.split} contains {len(records)} rows, expected {args.expected_rows}"
        )
    routed_records = [
        row for row in records if route_matches(row["messages"], args.route_kind)
    ]
    coverage = {
        "rows": len(routed_records),
        "per_category": dict(sorted(Counter(
            str(row["category"]) for row in routed_records
        ).items())),
        "per_source_model": dict(sorted(Counter(
            str(row["source_model"]) for row in routed_records
        ).items())),
    }

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_dir = args.adapter_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    llm = LLM(
        model=args.model,
        tokenizer=adapter_dir.as_posix(),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_lora=True,
        max_lora_rank=16,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    request = LoRARequest("liars-prompt-router", 1, adapter_dir.as_posix())

    control_prompts = render(
        tokenizer,
        records,
        load_prompt(args.control_config.resolve()),
        max_prompt_chars=args.max_prompt_chars,
        context_truncation=args.context_truncation,
    )
    started = time.perf_counter()
    control_outputs = llm.generate(
        control_prompts,
        sampling,
        lora_request=request,
    )
    control_elapsed = time.perf_counter() - started
    control_rows = evaluated_rows(records, control_prompts, control_outputs)

    specialist_prompts = render(
        tokenizer,
        routed_records,
        load_prompt(args.specialist_config.resolve()),
        max_prompt_chars=args.max_prompt_chars,
        context_truncation=args.context_truncation,
    )
    started = time.perf_counter()
    specialist_outputs = llm.generate(
        specialist_prompts,
        sampling,
        lora_request=request,
    )
    specialist_elapsed = time.perf_counter() - started
    specialist_rows = evaluated_rows(
        routed_records,
        specialist_prompts,
        specialist_outputs,
    )
    specialist_by_key = {
        (str(row["dataset"]), str(row["index"])): row for row in specialist_rows
    }
    routed_rows = []
    for row in control_rows:
        key = (str(row["dataset"]), str(row["index"]))
        selected = dict(specialist_by_key.get(key, row))
        selected["routed"] = key in specialist_by_key
        selected["route_kind"] = (
            args.route_kind if key in specialist_by_key else None
        )
        routed_rows.append(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("control", control_rows),
        (args.condition_name, routed_rows),
        (f"{args.condition_name}_specialist", specialist_rows),
    ):
        with (args.output_dir / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "split": args.split,
        "split_seed": args.split_seed,
        "rows": len(records),
        "conditions": {
            "control": summarize(control_rows, control_elapsed),
            args.condition_name: summarize(
                routed_rows,
                control_elapsed + specialist_elapsed,
            ),
        },
        "routes": {
            "control": {"rows": 0, "per_category": {}, "per_source_model": {}},
            args.condition_name: coverage,
        },
        "specialist_score_seconds": specialist_elapsed,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
