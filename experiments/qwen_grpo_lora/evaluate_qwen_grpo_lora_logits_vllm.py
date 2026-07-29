#!/usr/bin/env python3
"""Score GRPO LoRA adapters with constrained binary next-token vLLM logits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "qwen_grpo_lora"))

from evaluate_qwen_grpo_lora import (  # noqa: E402
    SplitRecords,
    cfg_get,
    cfg_get_default,
    load_split,
    load_training_config,
    macro_metrics,
    per_dataset_table,
)
from evaluate_qwen_grpo_lora_logits import (  # noqa: E402
    apply_global_limit,
    completion_prefix,
    label_strings,
    score_summary,
    single_token_id,
    threshold_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--prefix-variant",
        action="append",
        choices=["prediction", "prediction_space", "empty_reasoning", "empty_reasoning_space"],
        help="May be repeated; defaults to direct Prediction: and an empty reasoning scaffold.",
    )
    parser.add_argument("--label-style", choices=["plain", "space"], default="plain")
    parser.add_argument("--global-limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--disable-language-model-only", action="store_true")
    return parser.parse_args()


def load_evaluation_config(adapter_dir: Path) -> dict[str, Any]:
    """Load either a GRPO result config or a privileged-distillation YAML."""
    try:
        return load_training_config(adapter_dir)
    except FileNotFoundError:
        config_path = adapter_dir.parent / "config.yaml"
        if not config_path.is_file():
            raise
        config = yaml.safe_load(config_path.read_text()) or {}
        student = config["student"]
        return {
            "method": config["method"],
            "model": student["model"],
            "judge": {
                "prompt": student["prompt"],
                "enable_thinking": False,
                "max_prompt_chars": student["max_prompt_chars"],
                "context_truncation": student["context_truncation"],
                "include_reasoning": student["include_reasoning"],
                "reasoning_max_chars": student["reasoning_max_chars"],
                "reasoning_truncation": student["reasoning_truncation"],
            },
            "inference": {"prompt": student["prompt"]},
            "training": {"max_prompt_length": student["max_length"]},
            "source_config": config,
        }


def logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def binary_score(
    first_token_logprobs: dict[Any, Any],
    label0_id: int,
    label1_id: int,
) -> tuple[float, float]:
    values = {int(key): logprob_value(value) for key, value in first_token_logprobs.items()}
    missing = [token_id for token_id in (label0_id, label1_id) if token_id not in values]
    if missing:
        raise ValueError(f"vLLM omitted requested label token ids: {missing}")
    margin = values[label1_id] - values[label0_id]
    clipped = max(-80.0, min(80.0, margin))
    return 1.0 / (1.0 + math.exp(-clipped)), margin


def score_condition(
    *,
    llm: Any,
    sampling: Any,
    lora_request: Any,
    records: SplitRecords,
    prefix_variant: str,
    label0_id: int,
    label1_id: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    prefix = completion_prefix(prefix_variant)
    prompts = [prompt + prefix for prompt in records.frame["prompt"]]
    started = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    elapsed = time.time() - started

    rows = []
    for (_, row), output in zip(records.frame.iterrows(), outputs, strict=True):
        if not output.outputs or not output.outputs[0].logprobs:
            raise RuntimeError("vLLM returned no first-token logprobs")
        score, margin = binary_score(
            output.outputs[0].logprobs[0] or {},
            label0_id,
            label1_id,
        )
        rows.append({
            "dataset": row["dataset"],
            "index": row["index"],
            "label": int(row["label"]),
            "score": score,
            "logit_margin": margin,
            "generated_token": output.outputs[0].text,
        })
    return pd.DataFrame(rows), {
        "score_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed > 0 else float("nan"),
    }


def main() -> None:
    args = parse_args()
    adapter_dir = args.adapter_dir.resolve()
    training_config = load_evaluation_config(adapter_dir)
    variants = args.prefix_variant or ["prediction", "empty_reasoning"]
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else adapter_dir.parent / f"{args.split}_vllm_logits"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    from peft import PeftConfig
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_name = str(training_config["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label0, label1 = label_strings(args.label_style)
    label0_id = single_token_id(tokenizer, label0)
    label1_id = single_token_id(tokenizer, label1)
    include_reasoning = bool(cfg_get(training_config, "judge.include_reasoning"))
    inference_prompt = str(
        cfg_get_default(
            training_config,
            "inference.prompt",
            cfg_get(training_config, "judge.prompt"),
        )
    )
    records = load_split(
        args.split,
        args.splits_dir.resolve(),
        prompt_template=inference_prompt,
        tokenizer=tokenizer,
        max_prompt_chars=int(cfg_get(training_config, "judge.max_prompt_chars")),
        context_truncation=str(cfg_get(training_config, "judge.context_truncation")),
        include_reasoning=include_reasoning,
        reasoning_max_chars=int(cfg_get(training_config, "judge.reasoning_max_chars")),
        reasoning_truncation=str(cfg_get(training_config, "judge.reasoning_truncation")),
        enable_thinking=bool(cfg_get(training_config, "judge.enable_thinking")),
    )
    records = apply_global_limit(records, args.global_limit)
    max_model_len = args.max_model_len or (
        int(cfg_get(training_config, "training.max_prompt_length")) + 32
    )
    max_lora_rank = int(PeftConfig.from_pretrained(adapter_dir).r)
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "tokenizer": model_name,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_lora": True,
        "max_lora_rank": max_lora_rank,
        "max_model_len": max_model_len,
        "trust_remote_code": False,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if not args.disable_language_model_only:
        llm_kwargs.update(language_model_only=True, skip_mm_profiling=True)

    print(
        f"loading {model_name} with rank-{max_lora_rank} LoRA; "
        f"rows={len(records.frame)} variants={variants}",
        flush=True,
    )
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=2,
        logprob_token_ids=[label0_id, label1_id],
        allowed_token_ids=[label0_id, label1_id],
    )
    lora_request = LoRARequest(
        lora_name=adapter_dir.parent.name,
        lora_int_id=1,
        lora_path=adapter_dir.as_posix(),
    )

    summaries: dict[str, Any] = {}
    for variant in variants:
        predictions, timing = score_condition(
            llm=llm,
            sampling=sampling,
            lora_request=lora_request,
            records=records,
            prefix_variant=variant,
            label0_id=label0_id,
            label1_id=label1_id,
        )
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        grid = threshold_grid(predictions)
        best = grid.iloc[0].to_dict()
        metrics = macro_metrics(predictions, args.threshold)
        result = {
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": f"{training_config['method']}_vllm_logits_{variant}_{args.label_style}",
            "split": args.split,
            "n": int(len(predictions)),
            "metrics": metrics,
            "best_threshold": best,
            "score_summary": score_summary(predictions),
            "datasets_at_baseline": per_dataset_table(predictions, args.threshold),
            "timing": {
                **timing,
                "note": "scoring-only; excludes model startup and dataset preparation",
            },
            "config": {
                "adapter_dir": adapter_dir.as_posix(),
                "model": model_name,
                "prefix_variant": variant,
                "label_style": args.label_style,
                "label_ids": [label0_id, label1_id],
                "include_source_reasoning": include_reasoning,
                "inference_prompt": inference_prompt,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "training_config": training_config,
            },
        }
        predictions.to_csv(variant_dir / "predictions.csv", index=False)
        grid.to_csv(variant_dir / "threshold_grid.csv", index=False)
        (variant_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        summaries[variant] = {
            "metrics": metrics,
            "best_threshold": best,
            "timing": timing,
            "unique_scores": int(predictions["score"].nunique()),
        }
        print(json.dumps({variant: summaries[variant]}, indent=2), flush=True)

    (output_root / "summary.json").write_text(
        json.dumps({
            "adapter_dir": adapter_dir.as_posix(),
            "split": args.split,
            "variants": summaries,
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
