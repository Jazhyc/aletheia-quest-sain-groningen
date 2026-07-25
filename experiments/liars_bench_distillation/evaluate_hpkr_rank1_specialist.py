#!/usr/bin/env python3
"""Evaluate rank-1 HP-KR specialists in one shared Qwen/vLLM session."""

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

from experiments.liars_bench_distillation.evaluate_prompt_sweep import (
    load_prompt,
    split_records,
)
from experiments.liars_bench_distillation.evaluate_students import (
    balanced_accuracy,
    grouped_metrics,
    parse_prediction,
)
from experiments.liars_bench_hpkr_prompts.evaluate import PROMPTS as HPKR_PROMPTS
from experiments.privileged_information_distillation.core import build_student_prompt


CATEGORY = "harm-pressure-knowledge-report"


def render_prompts(
    tokenizer: Any,
    records: list[dict[str, Any]],
    prompt_template: str,
    *,
    max_prompt_chars: int,
) -> list[str]:
    """Render the frozen no-thinking specialist interface."""
    return [
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": build_student_prompt(
                    row["messages"],
                    prompt_template,
                    max_prompt_chars,
                    "head_tail",
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
    """Convert vLLM completions to the shared binary-generation schema."""
    rows = []
    for record, prompt, output in zip(records, prompts, outputs, strict=True):
        generation = output.outputs[0].text if output.outputs else ""
        prediction = parse_prediction(generation)
        rows.append({
            **{
                key: record[key]
                for key in ("dataset", "index", "category", "source_model", "label")
            },
            "prediction": 0 if prediction is None else prediction,
            "parse_error": prediction is None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "generation": generation,
        })
    return rows


def summarize(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """Summarize one balanced HP-KR condition."""
    return {
        "balanced_accuracy": balanced_accuracy(rows)["balanced_accuracy"],
        "metrics": balanced_accuracy(rows),
        "per_source_model": grouped_metrics(rows, "source_model"),
        "parse_errors": sum(bool(row["parse_error"]) for row in rows),
        "score_seconds": elapsed,
    }


def parse_adapter(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"invalid --adapter {value!r}; expected NAME=PATH")
    return name, Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--phoenix-adapter", type=Path, required=True)
    parser.add_argument("--phoenix-prompt-config", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "confirmation"),
        required=True,
    )
    parser.add_argument("--split-seed", type=int, default=20260725)
    parser.add_argument("--expected-rows", type=int, default=100)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    args = parser.parse_args()

    all_records = [
        json.loads(line)
        for line in args.eval_artifact.read_text().splitlines()
        if line.strip()
    ]
    records = [
        row
        for row in split_records(all_records, args.split, seed=args.split_seed)
        if row["category"] == CATEGORY
    ]
    if len(records) != args.expected_rows:
        raise ValueError(
            f"{args.split} contains {len(records)} HP-KR rows, "
            f"expected {args.expected_rows}"
        )
    label_counts = Counter(int(row["label"]) for row in records)
    if label_counts != Counter({0: 50, 1: 50}):
        raise ValueError(f"expected balanced labels, got {dict(label_counts)}")

    adapters = [parse_adapter(value) for value in args.adapter]
    names = [name for name, _ in adapters]
    if len(names) != len(set(names)):
        raise ValueError(f"adapter names must be unique, got {names}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_lora=True,
        max_lora_rank=16,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    epistemic_prompts = render_prompts(
        tokenizer,
        records,
        HPKR_PROMPTS["knowledge_report_type"],
        max_prompt_chars=args.max_prompt_chars,
    )
    phoenix_prompts = render_prompts(
        tokenizer,
        records,
        load_prompt(args.phoenix_prompt_config.resolve()),
        max_prompt_chars=args.max_prompt_chars,
    )

    conditions: list[tuple[str, list[str], Any]] = [
        ("base_epistemic", epistemic_prompts, None),
        (
            "phoenix_truthful_alternative",
            phoenix_prompts,
            LoRARequest(
                "phoenix-truthful-alternative",
                1,
                args.phoenix_adapter.resolve().as_posix(),
            ),
        ),
    ]
    for offset, (name, path) in enumerate(adapters, start=2):
        conditions.append((
            name,
            epistemic_prompts,
            LoRARequest(name, offset, path.as_posix()),
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "split": args.split,
        "split_seed": args.split_seed,
        "rows": len(records),
        "labels": dict(sorted(label_counts.items())),
        "conditions": {},
    }
    for name, prompts, request in conditions:
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, lora_request=request)
        elapsed = time.perf_counter() - started
        rows = evaluated_rows(records, prompts, outputs)
        with (args.output_dir / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result["conditions"][name] = summarize(rows, elapsed)
        print(name, json.dumps(result["conditions"][name]), flush=True)

    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
