#!/usr/bin/env python3
"""Score two Phoenix adapters on untouched Gemma soft-trigger rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_students import (
    balanced_accuracy,
    grouped_metrics,
    parse_prediction,
)
from experiments.privileged_information_distillation.core import build_student_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", action="append", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.eval_artifact.read_text().splitlines()]
    if len(records) != 1000 or {row["label"] for row in records} != {0, 1}:
        raise RuntimeError("expected the frozen balanced 1,000-row confirmation set")
    if any("gemma" not in row["source_model"].lower() for row in records):
        raise RuntimeError("confirmation artifact contains a non-Gemma source")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapters = []
    for value in args.adapter:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid adapter {value!r}")
        adapters.append((name, Path(path).resolve()))
    config = OmegaConf.load(ROOT / "configs/privileged_information_distillation.yaml")
    prompt_template = str(config.student.prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_student_prompt(
                row["messages"], prompt_template, 6000, "head_tail"
            )}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in records
    ]
    llm = LLM(
        model=args.model,
        tokenizer=adapters[0][1].as_posix(),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_lora=True,
        max_lora_rank=16,
    )
    sampling = SamplingParams(max_tokens=512, temperature=0.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"n": len(records), "conditions": {}}
    for adapter_index, (name, path) in enumerate(adapters, start=1):
        started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling,
            lora_request=LoRARequest(name, adapter_index, path.as_posix()),
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
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
