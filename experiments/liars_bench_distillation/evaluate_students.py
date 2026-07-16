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
from experiments.liars_bench_hpkr_prompts.evaluate import PROMPTS as HPKR_PROMPTS
from experiments.liars_bench_distillation.router import (
    is_action_report_conversation,
)


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
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument(
        "--context-truncation", choices=("head_tail", "tail"), default="head_tail"
    )
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
                row["messages"],
                prompt_template,
                args.max_prompt_chars,
                args.context_truncation,
            )}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in records
    ]
    hpkr_offsets = [
        offset for offset, row in enumerate(records)
        if row["category"] == "harm-pressure-knowledge-report"
    ]
    action_offsets = [
        offset for offset, row in enumerate(records)
        if is_action_report_conversation(row["messages"])
    ]
    action_counts: dict[str, int] = defaultdict(int)
    for offset in action_offsets:
        action_counts[str(records[offset]["category"])] += 1
    print(
        "action_router",
        json.dumps({"rows": len(action_offsets), "per_category": action_counts}),
        flush=True,
    )
    epistemic_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_student_prompt(
                records[offset]["messages"],
                HPKR_PROMPTS["knowledge_report_type"],
                args.max_prompt_chars,
                args.context_truncation,
            )}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for offset in hpkr_offsets
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
    result = {
        "rows": len(records),
        "conditions": {},
        "routed_conditions": {},
        "action_router": {
            "rows": len(action_offsets),
            "per_category": dict(sorted(action_counts.items())),
        },
        "action_routed_conditions": {},
        "content_routed_conditions": {},
    }
    general_rows: dict[str, list[dict[str, Any]]] = {}
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
        general_rows[name] = evaluated
        print(name, json.dumps(result["conditions"][name]), flush=True)
        specialist_started = time.perf_counter()
        specialist_outputs = llm.generate(
            epistemic_prompts,
            sampling,
            lora_request=LoRARequest(name, offset, path.as_posix()),
        )
        specialist_elapsed = time.perf_counter() - specialist_started
        routed = [dict(row) for row in evaluated]
        specialist_records = []
        for row_offset, output in zip(hpkr_offsets, specialist_outputs, strict=True):
            generation = output.outputs[0].text if output.outputs else ""
            parsed = parse_prediction(generation)
            prediction = 0 if parsed is None else parsed
            routed[row_offset]["prediction"] = prediction
            routed[row_offset]["parse_error"] = parsed is None
            specialist_records.append({
                **{key: records[row_offset][key] for key in (
                    "dataset", "index", "category", "source_model", "label"
                )},
                "prediction": prediction,
                "parse_error": parsed is None,
                "generation": generation,
            })
        with (args.output_dir / f"{name}_epistemic.jsonl").open("w") as handle:
            for row in specialist_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        routed_categories = grouped_metrics(routed, "category")
        result["routed_conditions"][name] = {
            "macro_category_balanced_accuracy": sum(
                metrics["balanced_accuracy"] for metrics in routed_categories.values()
            ) / len(routed_categories),
            "metrics": balanced_accuracy(routed),
            "per_category": routed_categories,
            "per_source_model": grouped_metrics(routed, "source_model"),
            "specialist_parse_errors": sum(
                row["parse_error"] for row in specialist_records
            ),
            "specialist_score_seconds": specialist_elapsed,
        }
        print(
            f"{name}_routed",
            json.dumps(result["routed_conditions"][name]),
            flush=True,
        )

    base_started = time.perf_counter()
    base_outputs = llm.generate(epistemic_prompts, sampling)
    base_elapsed = time.perf_counter() - base_started
    base_routed = [dict(row) for row in general_rows[adapters[0][0]]]
    for row_offset, output in zip(hpkr_offsets, base_outputs, strict=True):
        generation = output.outputs[0].text if output.outputs else ""
        parsed = parse_prediction(generation)
        base_routed[row_offset]["prediction"] = 0 if parsed is None else parsed
        base_routed[row_offset]["parse_error"] = parsed is None
    base_categories = grouped_metrics(base_routed, "category")
    result["routed_conditions"]["base_epistemic"] = {
        "general_member": adapters[0][0],
        "macro_category_balanced_accuracy": sum(
            metrics["balanced_accuracy"] for metrics in base_categories.values()
        ) / len(base_categories),
        "metrics": balanced_accuracy(base_routed),
        "per_category": base_categories,
        "per_source_model": grouped_metrics(base_routed, "source_model"),
        "specialist_score_seconds": base_elapsed,
    }
    print(
        "base_epistemic_routed",
        json.dumps(result["routed_conditions"]["base_epistemic"]),
        flush=True,
    )

    baseline_name = adapters[0][0]
    for name, _ in adapters:
        action_routed = [dict(row) for row in general_rows[baseline_name]]
        content_routed = [dict(row) for row in base_routed]
        for row_offset in action_offsets:
            action_routed[row_offset] = dict(general_rows[name][row_offset])
            content_routed[row_offset] = dict(general_rows[name][row_offset])
        for destination, routed in (
            (result["action_routed_conditions"], action_routed),
            (result["content_routed_conditions"], content_routed),
        ):
            categories = grouped_metrics(routed, "category")
            destination[name] = {
                "macro_category_balanced_accuracy": sum(
                    metrics["balanced_accuracy"] for metrics in categories.values()
                ) / len(categories),
                "metrics": balanced_accuracy(routed),
                "per_category": categories,
                "per_source_model": grouped_metrics(routed, "source_model"),
                "parse_errors": sum(row["parse_error"] for row in routed),
            }
        print(
            f"{name}_action_routed",
            json.dumps(result["action_routed_conditions"][name]),
            flush=True,
        )
        print(
            f"{name}_content_routed",
            json.dumps(result["content_routed_conditions"][name]),
            flush=True,
        )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
