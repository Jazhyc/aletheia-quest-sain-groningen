#!/usr/bin/env python3
"""Cache Q397 auxiliary label logits on a labeled public split."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml

from experiments.q397_readout_adaptation.run import (
    DEFAULT_ADAPTER,
    MODEL_ID,
    READOUTS,
    requested_logprobs,
    resolve_token_ids,
    scores_from_logprobs,
)


ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "dev_splits"
EXPECTED_ROWS = {"train": 6573, "validation": 822, "test": 821}


def family_from_text(value: str) -> str:
    """Return the public base-model family without using an organism id."""
    lowered = value.casefold()
    if "qwen" in lowered:
        return "Qwen"
    if "gemma" in lowered:
        return "Gemma"
    if "nemotron" in lowered or "nvidia" in lowered:
        return "Nemotron"
    raise ValueError(f"unknown base-model family in {value!r}")


def load_records(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load messages and local labels from a frozen public split manifest."""
    from datasets import load_dataset

    config_path = SPLITS_DIR / f"dry.{split}.yaml"
    config = yaml.safe_load(config_path.read_text())
    records: list[dict[str, Any]] = []
    for declaration in config["datasets"]:
        dataset_name = str(declaration["name"])
        dataset = load_dataset(dataset_name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        labels_path = Path(str(declaration["labels_uri"]))
        if not labels_path.is_absolute():
            labels_path = ROOT / labels_path
        with labels_path.open(newline="") as handle:
            labels = {
                str(row[str(declaration["id_column"])]): int(
                    str(row[str(declaration["label_column"])]).casefold() == "true"
                )
                for row in csv.DictReader(handle)
            }
        for row in dataset:
            index = str(row["index"])
            if index not in labels:
                continue
            records.append(
                {
                    "dataset": dataset_name,
                    "family": family_from_text(
                        str(row.get("model") or dataset_name)
                    ),
                    "index": row["index"],
                    "label": labels[index],
                    "messages": row["messages"],
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    expected = EXPECTED_ROWS[split]
    if len(records) != expected:
        raise ValueError(
            f"{split} manifests resolved to {len(records)} rows; expected {expected}"
        )
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Score one split in a single local-vLLM adapter pass."""
    from experiments.phoenix_verbalizer_sweep.run_ndif_base_verbalizers import (
        CONDITIONS,
        build_direct_prompt,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_dir = args.adapter_dir.resolve()
    records = load_records(args.split, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    pairs, union_ids = resolve_token_ids(tokenizer)
    condition = next(item for item in CONDITIONS if item.name == "digits_frozen")
    prompts = [
        build_direct_prompt(record["messages"], tokenizer, condition)
        for record in records
    ]

    llm = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=4096,
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(union_ids),
        logprob_token_ids=union_ids,
        allowed_token_ids=union_ids,
    )
    request = LoRARequest(adapter_dir.parent.name, 1, adapter_dir.as_posix())
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, lora_request=request)
    elapsed = time.perf_counter() - started
    logprobs = [requested_logprobs(output, union_ids) for output in outputs]
    scores = scores_from_logprobs(logprobs, pairs)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scores.jsonl").open("w") as handle:
        for position, record in enumerate(records):
            row = {
                "dataset": record["dataset"],
                "family": record["family"],
                "index": record["index"],
                "label": record["label"],
                "prompt_sha256": hashlib.sha256(
                    prompts[position].encode("utf-8")
                ).hexdigest(),
                "requested_logprobs": {
                    str(token_id): logprobs[position][token_id]
                    for token_id in union_ids
                },
                "scores": {
                    name: float(values[position]) for name, values in scores.items()
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "method": args.method,
        "split": args.split,
        "rows": len(records),
        "model": MODEL_ID,
        "adapter_dir": str(adapter_dir),
        "fixed_prompt_condition": asdict(condition),
        "readouts": [asdict(readout) for readout in READOUTS],
        "token_ids": {name: list(pair) for name, pair in pairs.items()},
        "adapter_seconds": elapsed,
        "families": {
            family: sum(record["family"] == family for record in records)
            for family in sorted({record["family"] for record in records})
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(EXPECTED_ROWS), default="train")
    parser.add_argument("--method", default="q397_family_calibration_train_cache_v1")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "results/blackbox" / args.method
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
