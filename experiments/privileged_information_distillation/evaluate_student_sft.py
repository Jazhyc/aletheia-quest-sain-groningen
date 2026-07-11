#!/usr/bin/env python3
"""Evaluate one or more privileged-distillation LoRA students with one vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import build_student_prompt
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import (
    binary_metrics,
    load_split_config,
    load_labels,
    macro_metrics,
)

PREDICTION_RE = re.compile(r"(?im)\bPrediction\s*:\s*<?([01])>?")
STRICT_RE = re.compile(
    r"(?is)^\s*<reasoning_summary>\s*(.+?)\s*</reasoning_summary>\s*"
    r"Prediction\s*:\s*([01])\s*$"
)
COUNTERFACTUAL_STRICT_RE = re.compile(
    r"(?is)^\s*<reasoning_summary>\s*(.+?)\s*</reasoning_summary>\s*"
    r"<facts>\s*(.+?)\s*</facts>\s*"
    r"<contradiction>\s*(.+?)\s*</contradiction>\s*"
    r"Prediction\s*:\s*([01])\s*$"
)
PREDICTION_ONLY_STRICT_RE = re.compile(r"(?is)^\s*Prediction\s*:\s*<?([01])>?\s*$")


def parse_prediction(text: str) -> int | None:
    matches = PREDICTION_RE.findall(text)
    return int(matches[-1]) if matches else None


def scenario_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result = {"all": macro_metrics(frame, 0.5)}
    for scenario in ("instructed", "varied"):
        subset = frame[frame["dataset"].str.contains(f"dev-{scenario}-deception")]
        if not subset.empty:
            result[scenario] = macro_metrics(subset, 0.5)
    return result


def load_retrieval_cache(path: Path | None) -> dict[tuple[str, Any], str]:
    if path is None:
        return {}
    references = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        passages = record.get("passages") or []
        text = "\n".join(
            f"- {passage.get('title', '')}: {passage.get('text', '')}"
            for passage in passages
            if passage.get("text")
        )
        references[(record["dataset"], record["index"])] = text
    return references


def load_records(
    split: str,
    splits_dir: Path,
    config: dict[str, Any],
    tokenizer: Any,
    references: dict[tuple[str, Any], str] | None = None,
) -> pd.DataFrame:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    for dataset_cfg in load_split_config(splits_dir / f"dry.{split}.yaml", ROOT):
        labels = load_labels(dataset_cfg)
        label_by_index = dict(zip(labels["index"], labels["label"], strict=True))
        dataset = load_dataset(dataset_cfg.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        found: set[Any] = set()
        for row in dataset:
            index = row["index"]
            if index not in label_by_index:
                continue
            raw_prompt = build_student_prompt(
                row["messages"],
                config["student"]["prompt"],
                int(config["student"]["max_prompt_chars"]),
                config["student"]["context_truncation"],
            )
            reference = (references or {}).get((dataset_cfg.name, index), "")
            if reference:
                raw_prompt += (
                    "\n\n<reference_material>\n"
                    "The following retrieved text may be incomplete or irrelevant. "
                    "Use it only when it clearly matches the question.\n"
                    f"{reference}\n</reference_material>"
                )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rows.append({
                "dataset": dataset_cfg.name,
                "index": index,
                "label": int(label_by_index[index]),
                "prompt": prompt,
            })
            found.add(index)
        if len(found) != len(labels):
            raise RuntimeError(f"{dataset_cfg.name}: found {len(found)}/{len(labels)} labeled rows")
    return pd.DataFrame(rows)


def evaluate_adapter(
    llm: Any,
    sampling: Any,
    records: pd.DataFrame,
    adapter_dir: Path,
    lora_id: int,
    strict_re: re.Pattern[str] = STRICT_RE,
) -> tuple[pd.DataFrame, float]:
    from vllm.lora.request import LoRARequest

    request = LoRARequest(adapter_dir.parent.name, lora_id, adapter_dir.as_posix())
    started = time.time()
    outputs = llm.generate(records["prompt"].tolist(), sampling, lora_request=request)
    elapsed = time.time() - started
    evaluated = records.drop(columns="prompt").copy()
    generations = [output.outputs[0].text if output.outputs else "" for output in outputs]
    predictions = [parse_prediction(text) for text in generations]
    evaluated["prediction"] = predictions
    evaluated["score"] = [float(value) if value is not None else 0.0 for value in predictions]
    evaluated["parse_error"] = [value is None for value in predictions]
    evaluated["format_valid"] = [strict_re.fullmatch(text) is not None for text in generations]
    evaluated["generation"] = generations
    return evaluated, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", action="append", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--retrieval-cache", type=Path)
    parser.add_argument("--run-name")
    args = parser.parse_args()

    adapter_dirs = [path.resolve() for path in args.adapter_dir]
    configs = [yaml.safe_load((path.parent / "config.yaml").read_text()) for path in adapter_dirs]
    first = configs[0]
    comparable = (
        "model", "prompt", "max_prompt_chars", "context_truncation",
        "target_format", "target_mode",
    )
    for config in configs[1:]:
        if any(config["student"].get(key) != first["student"].get(key) for key in comparable):
            raise SystemExit("student prompt/model settings differ across adapters")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(adapter_dirs[0])
    references = load_retrieval_cache(
        args.retrieval_cache.resolve() if args.retrieval_cache is not None else None
    )
    records = load_records(
        args.split,
        args.splits_dir.resolve(),
        first,
        tokenizer,
        references=references,
    )
    print(f"loaded {len(records)} rows across {records['dataset'].nunique()} datasets", flush=True)
    llm = LLM(
        model=first["student"]["model"],
        tokenizer=adapter_dirs[0].as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=max(int(config["student"]["lora"]["r"]) for config in configs),
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    if first["student"].get("target_mode") == "prediction_only":
        strict_re = PREDICTION_ONLY_STRICT_RE
    elif first["student"].get("target_format") == "counterfactual":
        strict_re = COUNTERFACTUAL_STRICT_RE
    else:
        strict_re = STRICT_RE

    for lora_id, (adapter_dir, config) in enumerate(zip(adapter_dirs, configs, strict=True), 1):
        evaluated, elapsed = evaluate_adapter(
            llm, sampling, records, adapter_dir, lora_id, strict_re=strict_re
        )
        method = config["method"]
        output_dir = adapter_dir.parent / (args.run_name or args.split)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluated.to_json(output_dir / "generations.jsonl", orient="records", lines=True)
        per_dataset = {}
        for dataset, group in evaluated.groupby("dataset", sort=True):
            per_dataset[dataset] = binary_metrics(
                group["label"].to_numpy(), group["score"].to_numpy(), 0.5
            )
        result = {
            "method": method,
            "split": args.split,
            "learning_rate": config["student"]["training"]["learning_rate"],
            "metrics": scenario_metrics(evaluated),
            "per_dataset": per_dataset,
            "parse_errors": int(evaluated["parse_error"].sum()),
            "format_valid": int(evaluated["format_valid"].sum()),
            "rows": len(evaluated),
            "score_seconds": elapsed,
            "rows_per_second": len(evaluated) / elapsed,
            "max_new_tokens": args.max_new_tokens,
            "retrieval_cache": (
                args.retrieval_cache.resolve().as_posix()
                if args.retrieval_cache is not None
                else None
            ),
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        metrics = result["metrics"]
        print(
            f"{method} lr={result['learning_rate']} all={metrics['all']} "
            f"instructed={metrics.get('instructed')} varied={metrics.get('varied')} "
            f"parse_errors={result['parse_errors']} time={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
