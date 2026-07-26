#!/usr/bin/env python3
"""Evaluate sparse Qwen-planned facts with frozen Phoenix direct-label margins."""

from __future__ import annotations

import argparse
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.evaluate_student_sft import (
    binary_token_ids,
    load_records,
    metrics_for_score,
    score_binary_prefixes,
)
from experiments.privileged_information_distillation.analyze_continuous_margins import (
    tie_metrics,
)
from experiments.wikidata_rag.evaluate_judge_sweep import format_passages


CONDITIONS = (
    "baseline",
    "recomputed_empty",
    "real_replace",
    "shuffled_replace",
    "real_blend",
    "shuffled_blend",
)


def load_cache(
    path: Path,
) -> tuple[
    dict[tuple[str, Any], str],
    dict[tuple[str, Any], str],
    set[tuple[str, Any]],
]:
    """Load sparse real and matched-shuffled references."""
    real: dict[tuple[str, Any], str] = {}
    shuffled: dict[tuple[str, Any], str] = {}
    active: set[tuple[str, Any]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["dataset"]), row["index"])
        real_text = format_passages(row.get("real_passages", []))
        shuffled_text = format_passages(row.get("shuffled_passages", []))
        if real_text:
            active.add(key)
            real[key] = real_text
            shuffled[key] = shuffled_text
    if any(not shuffled.get(key) for key in active):
        raise ValueError("every active real reference needs a matched shuffled reference")
    return real, shuffled, active


def mean_logodds(left: float, right: float) -> float:
    """Mean two probabilities in log-odds space."""
    epsilon = 1e-7
    left = min(1.0 - epsilon, max(epsilon, float(left)))
    right = min(1.0 - epsilon, max(epsilon, float(right)))
    value = 0.5 * (
        math.log(left / (1.0 - left)) + math.log(right / (1.0 - right))
    )
    return 1.0 / (1.0 + math.exp(-value))


def compose_conditions(
    baseline: pd.DataFrame,
    active_scores: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Copy cached scores for inactive rows and update only planner-active rows."""
    keys = ["dataset", "index"]
    base = baseline.set_index(keys).copy()
    active = active_scores.set_index(keys)
    if not active.index.is_unique or not base.index.is_unique:
        raise ValueError("duplicate dataset/index keys")
    if not active.index.isin(base.index).all():
        raise ValueError("active score keys are not a subset of the baseline")
    if not active["label"].equals(base.loc[active.index, "label"]):
        raise ValueError("active labels differ from the baseline")

    output = {}
    for condition in CONDITIONS:
        frame = base.copy()
        if condition == "recomputed_empty":
            frame.loc[active.index, "score"] = active["empty_score"]
        elif condition == "real_replace":
            frame.loc[active.index, "score"] = active["real_score"]
        elif condition == "shuffled_replace":
            frame.loc[active.index, "score"] = active["shuffled_score"]
        elif condition == "real_blend":
            frame.loc[active.index, "score"] = [
                mean_logodds(old, new)
                for old, new in zip(
                    base.loc[active.index, "score"],
                    active["real_score"],
                    strict=True,
                )
            ]
        elif condition == "shuffled_blend":
            frame.loc[active.index, "score"] = [
                mean_logodds(old, new)
                for old, new in zip(
                    base.loc[active.index, "score"],
                    active["shuffled_score"],
                    strict=True,
                )
            ]
        output[condition] = frame.reset_index()
    return output


def summarize_condition(frame: pd.DataFrame) -> dict[str, Any]:
    """Report macro/scenario metrics, per-dataset AUROC, and score ties."""
    metrics = metrics_for_score(frame, "score")
    per_dataset = {}
    for dataset, group in frame.groupby("dataset", sort=True):
        labels = group["label"].to_numpy()
        scores = group["score"].to_numpy()
        if len(np.unique(labels)) < 2:
            auroc = None
        else:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(labels, scores))
        per_dataset[str(dataset)] = {
            "auroc": auroc,
            "rows": len(group),
            "unique_scores": int(group["score"].nunique()),
        }
    return {
        "metrics": metrics,
        "per_dataset": per_dataset,
        "unique_scores": int(frame["score"].nunique()),
        "ties": tie_metrics(frame, "score"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline-generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.resolve()
    config = yaml.safe_load((adapter_dir.parent / "config.yaml").read_text())
    real_references, shuffled_references, active_keys = load_cache(args.cache.resolve())

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    ordinary = load_records(
        args.split,
        args.splits_dir.resolve(),
        config,
        tokenizer,
    )
    ordinary["key"] = list(zip(ordinary["dataset"], ordinary["index"], strict=True))
    active = ordinary[ordinary["key"].isin(active_keys)].copy()
    if len(active) != len(active_keys):
        missing = active_keys - set(active["key"])
        raise RuntimeError(f"planner cache has {len(missing)} unknown active keys")

    # Re-render the raw student prompts because references must be inside the user turn,
    # before the assistant generation marker.
    real_records = load_records(
        args.split,
        args.splits_dir.resolve(),
        config,
        tokenizer,
        references=real_references,
    )
    shuffled_records = load_records(
        args.split,
        args.splits_dir.resolve(),
        config,
        tokenizer,
        references=shuffled_references,
    )
    real_records["key"] = list(zip(
        real_records["dataset"], real_records["index"], strict=True
    ))
    shuffled_records["key"] = list(zip(
        shuffled_records["dataset"], shuffled_records["index"], strict=True
    ))
    real_active = real_records[real_records["key"].isin(active_keys)].set_index("key")
    shuffled_active = shuffled_records[
        shuffled_records["key"].isin(active_keys)
    ].set_index("key")
    active = active.set_index("key").sort_index()
    real_active = real_active.reindex(active.index)
    shuffled_active = shuffled_active.reindex(active.index)

    llm = LLM(
        model=config["student"]["model"],
        tokenizer=adapter_dir.as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=int(config["student"]["lora"]["r"]),
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
    )
    binary_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(binary_ids),
        logprob_token_ids=binary_ids,
        allowed_token_ids=binary_ids,
    )
    request = LoRARequest(adapter_dir.parent.name, 1, adapter_dir.as_posix())
    condition_prompts = [
        *[prompt + "Prediction:" for prompt in active["prompt"]],
        *[prompt + "Prediction:" for prompt in real_active["prompt"]],
        *[prompt + "Prediction:" for prompt in shuffled_active["prompt"]],
    ]
    started = time.time()
    values, missing_logits, score_seconds = score_binary_prefixes(
        llm,
        condition_prompts,
        sampling,
        request,
        binary_ids,
    )
    count = len(active)
    active_scores = active[["dataset", "index", "label"]].copy()
    active_scores["empty_score"] = values[:count]
    active_scores["real_score"] = values[count:2 * count]
    active_scores["shuffled_score"] = values[2 * count:]

    baseline = pd.read_json(args.baseline_generations, lines=True)[
        ["dataset", "index", "label", "direct_margin_score"]
    ].rename(columns={"direct_margin_score": "score"})
    conditions = compose_conditions(baseline, active_scores.reset_index(drop=True))
    result = {
        "split": args.split,
        "active_rows": count,
        "conditions": {
            name: summarize_condition(frame)
            for name, frame in conditions.items()
        },
        "active_score_deltas": {
            column: {
                "all": float((active_scores[column] - active_scores["empty_score"]).mean()),
                "honest": float((
                    active_scores.loc[active_scores["label"].eq(0), column]
                    - active_scores.loc[active_scores["label"].eq(0), "empty_score"]
                ).mean()),
                "deceptive": float((
                    active_scores.loc[active_scores["label"].eq(1), column]
                    - active_scores.loc[active_scores["label"].eq(1), "empty_score"]
                ).mean()),
            }
            for column in ("real_score", "shuffled_score")
        },
        "missing_logits": missing_logits,
        "score_seconds": score_seconds,
        "wall_seconds": time.time() - started,
        "cache": args.cache.resolve().as_posix(),
        "baseline_generations": args.baseline_generations.resolve().as_posix(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    active_scores.reset_index(drop=True).to_json(
        args.output_dir / "active_scores.jsonl", orient="records", lines=True
    )
    for name, frame in conditions.items():
        frame.assign(condition=name).to_json(
            args.output_dir / f"{name}.jsonl", orient="records", lines=True
        )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
