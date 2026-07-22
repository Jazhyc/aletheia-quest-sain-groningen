#!/usr/bin/env python3
"""Sweep concise reasoning-summary prompts for the deployed Phoenix members."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.evaluate_members import (  # noqa: E402
    evaluate_member,
)
from experiments.privileged_information_distillation.evaluate_student_sft import (  # noqa: E402
    load_records,
    output_mode_for_config,
    scenario_metrics,
    strict_pattern_for_config,
    vllm_max_lora_rank,
)


DEFAULT_MEMBERS = {
    "deception": ROOT
    / "results/blackbox/qwen9b_pid_specialist_material_rank1_v1/adapter",
    "resolved_intent": ROOT
    / "results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/adapter",
}

CONDITIONS = {
    "baseline": None,
    "sentence32": (
        "Keep the reasoning summary to one sentence of at most 32 words. "
        "State only the decisive evidence and do not restate the conversation."
    ),
    "sentence20": (
        "Keep the reasoning summary to one sentence of at most 20 words. "
        "State only the decisive evidence and do not restate the conversation."
    ),
    "clause12": (
        "Use one compressed reasoning clause of at most 12 words. Fragments are "
        "allowed; include only the decisive evidence."
    ),
    "contrast24": {
        "deception": (
            "Use at most 24 words in the reasoning summary: identify the decisive "
            "claim and its correction or confirmation. Do not restate the question."
        ),
        "resolved_intent": (
            "Use at most 24 words in the reasoning summary: state the final operative "
            "decision and whether the output executed it. Do not restate the trace."
        ),
    },
}


def parse_member(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"members must be NAME=ADAPTER_DIR, got {value!r}")
    return name, Path(path).resolve()


def prompt_for_condition(base_prompt: str, member: str, condition: str) -> str:
    """Insert a concision requirement without changing output order or schema."""
    instruction = CONDITIONS[condition]
    if instruction is None:
        return base_prompt
    if isinstance(instruction, dict):
        instruction = instruction[member]
    marker = "Output exactly:"
    if marker not in base_prompt:
        raise ValueError("student prompt is missing the output-schema marker")
    return base_prompt.replace(marker, f"{instruction}\n\n{marker}", 1)


def token_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    values = sorted(
        int(value)
        for value in frame.loc[~frame["routed_no_reasoning"], "generation_tokens"]
    )
    return {
        "active_rows": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": values[max(0, int(0.95 * len(values)) - 1)],
        "max": max(values),
        "cap_hits": sum(value >= 256 for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--condition", action="append", choices=tuple(CONDITIONS), default=[]
    )
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    members = (
        [parse_member(value) for value in args.member]
        if args.member
        else list(DEFAULT_MEMBERS.items())
    )
    conditions = args.condition or list(CONDITIONS)
    configs = {
        name: yaml.safe_load((adapter_dir.parent / "config.yaml").read_text())
        for name, adapter_dir in members
    }
    models = {config["student"]["model"] for config in configs.values()}
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
        max_lora_rank=vllm_max_lora_rank(
            [int(configs[name]["student"]["lora"]["r"]) for name, _ in members]
        ),
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    summary: dict[str, Any] = {
        "split": args.split,
        "max_new_tokens": args.max_new_tokens,
        "conditions": {},
    }
    lora_ids = {name: position for position, (name, _) in enumerate(members, 1)}
    for condition in conditions:
        summary["conditions"][condition] = {}
        for member, adapter_dir in members:
            config = copy.deepcopy(configs[member])
            config["student"]["prompt"] = prompt_for_condition(
                config["student"]["prompt"], member, condition
            )
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
                lora_ids[member],
                config,
                requires_reasoning=member == "resolved_intent",
            )
            elapsed = time.time() - started
            member_dir = args.output_dir / condition / member
            member_dir.mkdir(parents=True, exist_ok=True)
            evaluated.to_json(
                member_dir / "generations.jsonl", orient="records", lines=True
            )
            result = {
                "condition": condition,
                "member": member,
                "metrics": scenario_metrics(evaluated),
                "tokens": token_summary(evaluated),
                "parse_errors": int(evaluated["parse_error"].sum()),
                "format_valid": int(evaluated["format_valid"].sum()),
                "score_seconds": float(timing["generation_seconds"]),
                "wall_seconds": elapsed,
            }
            (member_dir / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            summary["conditions"][condition][member] = result
            print(json.dumps(result, sort_keys=True), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "member_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
