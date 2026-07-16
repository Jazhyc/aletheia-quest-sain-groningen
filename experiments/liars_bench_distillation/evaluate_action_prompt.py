#!/usr/bin/env python3
"""Evaluate a frozen action-report specialist prompt with several LoRAs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_students import (
    balanced_accuracy,
    grouped_metrics,
    parse_prediction,
)
from experiments.liars_bench_distillation.prompts import ACTION_REPORT_PROMPT
from experiments.liars_bench_distillation.router import is_action_report_conversation
from experiments.privileged_information_distillation.core import build_student_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", action="append", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-prompt-chars", type=int, default=4500)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    all_records = [
        json.loads(line) for line in args.eval_artifact.read_text().splitlines()
    ]
    records = [
        row for row in all_records if is_action_report_conversation(row["messages"])
    ]
    if not records:
        raise RuntimeError("the action-report router selected no rows")
    adapters = []
    for value in args.adapter:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --adapter {value!r}; expected NAME=PATH")
        adapters.append((name, Path(raw_path).resolve()))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_student_prompt(
                row["messages"],
                ACTION_REPORT_PROMPT,
                args.max_prompt_chars,
                "head_tail",
            )}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in records
    ]
    token_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    if max(token_lengths) > 2048:
        raise RuntimeError(
            f"action prompt exceeds deployment input cap: max={max(token_lengths)}"
        )

    llm = LLM(
        model=args.model,
        tokenizer=adapters[0][1].as_posix(),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_lora=True,
        max_lora_rank=16,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "rows": len(records),
        "max_prompt_chars": args.max_prompt_chars,
        "max_prompt_tokens": max(token_lengths),
        "conditions": {},
    }
    for offset, (name, path) in enumerate(adapters, start=1):
        started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling,
            lora_request=LoRARequest(name, offset, path.as_posix()),
        )
        elapsed = time.perf_counter() - started
        evaluated = []
        for row, output in zip(records, outputs, strict=True):
            generation = output.outputs[0].text if output.outputs else ""
            parsed = parse_prediction(generation)
            evaluated.append({
                **{key: row[key] for key in (
                    "dataset", "index", "category", "source_model", "label"
                )},
                "prediction": 0 if parsed is None else parsed,
                "parse_error": parsed is None,
                "generation": generation,
            })
        with (args.output_dir / f"{name}.jsonl").open("w") as handle:
            for row in evaluated:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result["conditions"][name] = {
            "metrics": balanced_accuracy(evaluated),
            "per_source_model": grouped_metrics(evaluated, "source_model"),
            "parse_errors": sum(row["parse_error"] for row in evaluated),
            "score_seconds": elapsed,
        }
        print(name, json.dumps(result["conditions"][name]), flush=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
