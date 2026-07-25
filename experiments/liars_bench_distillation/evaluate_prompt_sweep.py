#!/usr/bin/env python3
"""Evaluate general deception-judge prompts on a frozen Liars' Bench split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_students import (
    balanced_accuracy,
    category_source_metrics,
    grouped_metrics,
    parse_prediction,
)
from experiments.privileged_information_distillation.core import build_student_prompt


def parse_named_path(value: str) -> tuple[str, Path]:
    """Parse a NAME=PATH command-line value."""
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return name, path.resolve()


def split_records(
    records: list[dict[str, Any]],
    split: str,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Split every category/label cell deterministically into equal halves."""
    if split == "all":
        return list(records)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["category"]), int(row["label"]))].append(row)
    selected_keys: set[tuple[str, str]] = set()
    for rows in grouped.values():
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}\0{row['dataset']}\0{row['index']}"
                ).encode("utf-8")
            ).digest(),
        )
        development_count = len(ordered) // 2
        chosen = (
            ordered[:development_count]
            if split == "development"
            else ordered[development_count:]
        )
        selected_keys.update(
            (str(row["dataset"]), str(row["index"])) for row in chosen
        )
    return [
        row
        for row in records
        if (str(row["dataset"]), str(row["index"])) in selected_keys
    ]


def summarize(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """Compute frozen aggregate and grouped metrics for one prompt."""
    per_category = grouped_metrics(rows, "category")
    category_scores = [
        metrics["balanced_accuracy"]
        for metrics in per_category.values()
        if metrics["balanced_accuracy"] is not None
    ]
    return {
        "macro_category_balanced_accuracy": (
            sum(category_scores) / len(category_scores)
        ),
        "metrics": balanced_accuracy(rows),
        "per_category": per_category,
        "per_source_model": grouped_metrics(rows, "source_model"),
        "per_category_source_model": category_source_metrics(rows),
        "parse_errors": sum(bool(row["parse_error"]) for row in rows),
        "score_seconds": elapsed,
    }


def load_prompt(path: Path) -> str:
    """Load the student prompt from a small Hydra-compatible config."""
    config = yaml.safe_load(path.read_text())
    prompt = config.get("student", {}).get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path}: student.prompt must be a non-empty string")
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True, type=parse_named_path)
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

    names = [name for name, _ in args.prompt]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate prompt names: {names}")
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
    request = LoRARequest("liars-prompt-sweep", 1, adapter_dir.as_posix())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "split": args.split,
        "split_seed": args.split_seed,
        "rows": len(records),
        "conditions": {},
    }
    for name, config_path in args.prompt:
        prompt_template = load_prompt(config_path)
        rendered = [
            tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": build_student_prompt(
                        row["messages"],
                        prompt_template,
                        args.max_prompt_chars,
                        args.context_truncation,
                    ),
                }],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in records
        ]
        started = time.perf_counter()
        outputs = llm.generate(rendered, sampling, lora_request=request)
        elapsed = time.perf_counter() - started
        evaluated = []
        for row, raw_prompt, output in zip(records, rendered, outputs, strict=True):
            generation = output.outputs[0].text if output.outputs else ""
            parsed = parse_prediction(generation)
            evaluated.append({
                **{
                    key: row[key]
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
                "prompt_sha256": hashlib.sha256(
                    raw_prompt.encode("utf-8")
                ).hexdigest(),
                "generation": generation,
            })
        output_path = args.output_dir / f"{name}.jsonl"
        with output_path.open("w") as handle:
            for row in evaluated:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = summarize(evaluated, elapsed)
        summary["config_path"] = config_path.relative_to(ROOT).as_posix()
        result["conditions"][name] = summary
        print(name, json.dumps(summary), flush=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
