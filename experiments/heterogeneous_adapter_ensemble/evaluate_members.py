#!/usr/bin/env python3
"""Evaluate heterogeneous Qwen LoRA judges in one shared vLLM session."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.evaluate_student_sft import (
    evaluate_adapter,
    load_records,
    output_mode_for_config,
    scenario_metrics,
    strict_pattern_for_config,
    vllm_max_lora_rank,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import binary_metrics


def parse_member(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"members must be NAME=ADAPTER_DIR, got {value!r}")
    return name, Path(path).resolve()


def routed_no_reasoning_rows(records: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic zero-risk rows when the intent input is unavailable."""
    routed = records.drop(columns="prompt").copy()
    routed["prompt_sha256"] = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in records["prompt"]
    ]
    routed["finish_reason"] = "routed_no_reasoning"
    routed["generation_tokens"] = 0
    routed["prediction"] = 0
    routed["score"] = 0.0
    routed["parse_error"] = False
    routed["format_valid"] = True
    routed["generation"] = (
        "<reasoning_summary>Assistant reasoning unavailable; intent member "
        "not queried.</reasoning_summary>\nPrediction:0"
    )
    routed["routed_no_reasoning"] = True
    return routed


def evaluate_member(
    llm: Any,
    sampling: Any,
    records: pd.DataFrame,
    adapter_dir: Path,
    lora_id: int,
    config: dict[str, Any],
    *,
    requires_reasoning: bool,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if requires_reasoning:
        active = records[records["reasoning_present"]].reset_index(drop=True)
        inactive = records[~records["reasoning_present"]].reset_index(drop=True)
    else:
        active = records
        inactive = records.iloc[0:0]
    evaluated, timing = evaluate_adapter(
        llm,
        sampling,
        active,
        adapter_dir,
        lora_id,
        strict_re=strict_pattern_for_config(config),
        output_mode=output_mode_for_config(config),
    )
    evaluated["routed_no_reasoning"] = False
    if not inactive.empty:
        evaluated = pd.concat(
            [evaluated, routed_no_reasoning_rows(inactive)], ignore_index=True
        )
    return evaluated.sort_values(["dataset", "index"]).reset_index(drop=True), timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--requires-reasoning", action="append", default=[])
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    members = [parse_member(value) for value in args.member]
    names = [name for name, _ in members]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate member names: {names}")
    unknown = set(args.requires_reasoning) - set(names)
    if unknown:
        raise ValueError(f"unknown reasoning-routed members: {sorted(unknown)}")
    configs = [
        yaml.safe_load((adapter_dir.parent / "config.yaml").read_text())
        for _, adapter_dir in members
    ]
    models = {config["student"]["model"] for config in configs}
    if len(models) != 1:
        raise ValueError(f"member base models differ: {sorted(models)}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(members[0][1])
    llm = LLM(
        model=models.pop(),
        tokenizer=members[0][1].as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=vllm_max_lora_rank([
            int(config["student"]["lora"]["r"]) for config in configs
        ]),
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    for lora_id, ((name, adapter_dir), config) in enumerate(
        zip(members, configs, strict=True), 1
    ):
        records = load_records(
            args.split,
            args.splits_dir.resolve(),
            config,
            tokenizer,
        )
        started = time.time()
        evaluated, timing = evaluate_member(
            llm,
            sampling,
            records,
            adapter_dir,
            lora_id,
            config,
            requires_reasoning=name in args.requires_reasoning,
        )
        elapsed = time.time() - started
        output_dir = adapter_dir.parent / args.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluated.to_json(
            output_dir / "generations.jsonl", orient="records", lines=True
        )
        result = {
            "method": config["method"],
            "member": name,
            "split": args.split,
            "metrics": scenario_metrics(evaluated),
            "per_dataset": {
                dataset: binary_metrics(
                    group["label"].to_numpy(), group["score"].to_numpy(), 0.5
                )
                for dataset, group in evaluated.groupby("dataset", sort=True)
            },
            "rows": len(evaluated),
            "active_rows": int((~evaluated["routed_no_reasoning"]).sum()),
            "routed_no_reasoning": int(evaluated["routed_no_reasoning"].sum()),
            "parse_errors": int(evaluated["parse_error"].sum()),
            "format_valid": int(evaluated["format_valid"].sum()),
            "score_seconds": float(timing["generation_seconds"]),
            "wall_seconds": elapsed,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
