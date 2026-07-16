#!/usr/bin/env python3
"""Evaluate several distilled Qwen adapters on disjoint Liars' Bench rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import build_student_prompt


PREDICTION_RE = re.compile(r"(?im)\bPrediction\s*:\s*<?([01])>?\b")


def parse_prediction(text: str) -> int | None:
    matches = PREDICTION_RE.findall(text)
    return int(matches[-1]) if matches else None


def balanced_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = sum(row["label"] == row["prediction"] == 1 for row in rows)
    fn = sum(row["label"] == 1 and row["prediction"] == 0 for row in rows)
    fp = sum(row["label"] == 0 and row["prediction"] == 1 for row in rows)
    tn = sum(row["label"] == row["prediction"] == 0 for row in rows)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return {"balanced_accuracy": (recall + 1 - fpr) / 2, "recall": recall, "fpr": fpr}


def grouped_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: balanced_accuracy(group) for name, group in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter", action="append", required=True,
        help="Condition in NAME=ADAPTER_PATH form.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    records = [json.loads(line) for line in args.eval_artifact.read_text().splitlines()]
    config = OmegaConf.load(ROOT / "configs/privileged_information_distillation.yaml")
    prompt_template = str(config.student.prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_student_prompt(
                row["messages"], prompt_template, 3000, "tail"
            )}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in records
    ]
    adapters = []
    for value in args.adapter:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --adapter {value!r}; expected NAME=PATH")
        adapters.append((name, Path(path).resolve()))
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
    result = {"rows": len(records), "conditions": {}}
    for offset, (name, path) in enumerate(adapters, start=1):
        started = time.perf_counter()
        outputs = llm.generate(
            prompts, sampling,
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
        per_category = grouped_metrics(evaluated, "category")
        result["conditions"][name] = {
            "macro_category_balanced_accuracy": sum(
                metrics["balanced_accuracy"] for metrics in per_category.values()
            ) / len(per_category),
            "metrics": balanced_accuracy(evaluated),
            "per_category": per_category,
            "per_source_model": grouped_metrics(evaluated, "source_model"),
            "parse_errors": sum(row["parse_error"] for row in evaluated),
            "score_seconds": elapsed,
        }
        print(name, json.dumps(result["conditions"][name]), flush=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
