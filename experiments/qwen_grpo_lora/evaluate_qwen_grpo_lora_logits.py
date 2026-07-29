#!/usr/bin/env python3
"""Evaluate a binary Qwen GRPO LoRA adapter with next-token label logits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--global-limit", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument(
        "--reasoning-prefixes",
        type=Path,
        help=(
            "Optional generations.jsonl containing populated reasoning. When set, "
            "each row is scored after its cached prefix up through `Prediction:`."
        ),
    )
    parser.add_argument(
        "--prefix-variant",
        choices=["prediction", "prediction_space", "empty_reasoning", "empty_reasoning_space"],
        default="empty_reasoning",
        help="Assistant completion prefix placed before scoring the next label token.",
    )
    parser.add_argument(
        "--label-style",
        choices=["plain", "space"],
        default="space",
        help="Score 0/1 or space-prefixed 0/1 as the next token.",
    )
    parser.add_argument(
        "--exclude-reasoning",
        action="store_true",
        help=(
            "Do not expose the source model's assistant reasoning, even when the "
            "saved training config enabled it. Use this for leaderboard-aligned "
            "output-only evaluation."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def apply_global_limit(records: SplitRecords, limit: int | None) -> SplitRecords:
    if limit is None:
        return records
    return SplitRecords(
        frame=records.frame.iloc[:limit].reset_index(drop=True),
        dataset_names=records.dataset_names,
    )


def completion_prefix(variant: str) -> str:
    if variant == "prediction":
        return "Prediction:"
    if variant == "prediction_space":
        return "Prediction: "
    if variant == "empty_reasoning":
        return "<reasoning>\n</reasoning>\nPrediction:"
    if variant == "empty_reasoning_space":
        return "<reasoning>\n</reasoning>\nPrediction: "
    raise ValueError(f"unknown prefix variant {variant!r}")


def label_strings(style: str) -> tuple[str, str]:
    if style == "plain":
        return "0", "1"
    if style == "space":
        return " 0", " 1"
    raise ValueError(f"unknown label style {style!r}")


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return int(ids[0])


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {
        "balanced_accuracy": (recall + (1.0 - fpr)) / 2.0,
        "recall": recall,
        "fpr": fpr,
    }


def threshold_grid(frame: pd.DataFrame) -> pd.DataFrame:
    scores = frame["score"].to_numpy(dtype=float)
    candidates = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 501),
        scores,
        np.nextafter(scores, -np.inf),
        np.nextafter(scores, np.inf),
    ]))
    grouped = [
        (
            group["label"].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
        )
        for _, group in frame.groupby("dataset", sort=True)
    ]
    auroc = macro_metrics(frame, 0.5)["auroc"]
    rows = []
    for threshold in candidates:
        balanced_accuracies = []
        recalls = []
        fprs = []
        for labels, group_scores in grouped:
            metrics = threshold_metrics(labels, group_scores, float(threshold))
            recall = metrics["recall"]
            fpr = metrics["fpr"]
            if not np.isnan(recall):
                recalls.append(recall)
            if not np.isnan(fpr):
                fprs.append(fpr)
            if not np.isnan(metrics["balanced_accuracy"]):
                balanced_accuracies.append(metrics["balanced_accuracy"])
        rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": (
                float(np.mean(balanced_accuracies))
                if balanced_accuracies
                else None
            ),
            "auroc": auroc,
            "recall": float(np.mean(recalls)) if recalls else None,
            "fpr": float(np.mean(fprs)) if fprs else None,
        })
    grid = pd.DataFrame(rows)
    grid = grid.sort_values(
        ["balanced_accuracy", "auroc", "recall"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return grid


def evaluate_logits(
    *,
    model: Any,
    tokenizer: Any,
    records: SplitRecords,
    batch_size: int,
    prefix: str,
    label0_id: int,
    label1_id: int,
    max_model_len: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch

    rows = []
    started = time.time()
    model.eval()
    old_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        with torch.inference_mode():
            for start in range(0, len(records.frame), batch_size):
                batch = records.frame.iloc[start:start + batch_size]
                prompts = [
                    row["prompt"] + row.get("completion_prefix", prefix)
                    for _, row in batch.iterrows()
                ]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_model_len,
                ).to(model.device)
                outputs = model(**encoded)
                next_logits = outputs.logits[:, -1, [label0_id, label1_id]].float()
                probs = torch.softmax(next_logits, dim=-1)
                score = probs[:, 1].detach().cpu().numpy()
                margin = (
                    next_logits[:, 1] - next_logits[:, 0]
                ).detach().cpu().numpy()
                for (_, row), s, m in zip(batch.iterrows(), score, margin, strict=True):
                    rows.append({
                        "dataset": row["dataset"],
                        "index": row["index"],
                        "label": int(row["label"]),
                        "score": float(s),
                        "logit_margin": float(m),
                        "completion_prefix_source": row.get("completion_prefix_source", "constant"),
                    })
    finally:
        tokenizer.truncation_side = old_truncation_side

    elapsed = time.time() - started
    return pd.DataFrame(rows), {
        "score_time_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed > 0 else None,
    }


def score_summary(frame: pd.DataFrame) -> dict[str, Any]:
    scores = frame["score"].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "min": float(np.min(scores)),
        "p01": float(np.quantile(scores, 0.01)),
        "p05": float(np.quantile(scores, 0.05)),
        "p50": float(np.quantile(scores, 0.50)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
        "frac_lt_1e-3": float(np.mean(scores < 1e-3)),
        "frac_gt_0_999": float(np.mean(scores > 0.999)),
    }
    for label in [0, 1]:
        part = frame.loc[frame["label"] == label, "score"].to_numpy(dtype=float)
        if part.size:
            out[f"label_{label}"] = {
                "p05": float(np.quantile(part, 0.05)),
                "p50": float(np.quantile(part, 0.50)),
                "p95": float(np.quantile(part, 0.95)),
                "frac_lt_1e-3": float(np.mean(part < 1e-3)),
                "frac_gt_0_999": float(np.mean(part > 0.999)),
            }
    return out


def generated_completion_prefix(generation: str, fallback_prefix: str) -> tuple[str, str]:
    """Return generated text up through the final Prediction separator."""
    match = re.match(r"(?is)^(?P<prefix>.*?\bPrediction:\s*)(?P<label>[01])\s*$", generation.strip())
    if match is not None:
        return match.group("prefix"), "generated"
    text = generation.rstrip()
    if text:
        return text + "\n" + fallback_prefix, "fallback_after_generation"
    return fallback_prefix, "fallback_empty"


def attach_reasoning_prefixes(records: SplitRecords, path: Path, fallback_prefix: str) -> SplitRecords:
    by_key: dict[tuple[str, Any], tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["dataset"]), row["index"])
        by_key[key] = generated_completion_prefix(str(row.get("generation", "")), fallback_prefix)

    frame = records.frame.copy()
    prefixes = []
    sources = []
    missing = 0
    for _, row in frame.iterrows():
        prefix_source = by_key.get((str(row["dataset"]), row["index"]))
        if prefix_source is None:
            missing += 1
            prefix_source = (fallback_prefix, "missing")
        prefix, source = prefix_source
        prefixes.append(prefix)
        sources.append(source)
    if missing:
        raise RuntimeError(f"{path}: missing reasoning prefixes for {missing} rows")
    frame["completion_prefix"] = prefixes
    frame["completion_prefix_source"] = sources
    return SplitRecords(frame=frame, dataset_names=records.dataset_names)


def main() -> None:
    args = parse_args()
    adapter_dir = args.adapter_dir.resolve()
    training_config = load_training_config(adapter_dir)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else adapter_dir.parent / f"{args.split}_logits_{args.prefix_variant}_{args.label_style}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label0, label1 = label_strings(args.label_style)
    label0_id = single_token_id(tokenizer, label0)
    label1_id = single_token_id(tokenizer, label1)
    prefix = completion_prefix(args.prefix_variant)
    print(
        f"label0={label0!r}:{label0_id} label1={label1!r}:{label1_id} "
        f"prefix_variant={args.prefix_variant}",
        flush=True,
    )

    print(f"loading {args.split} split", flush=True)
    include_reasoning = (
        bool(cfg_get(training_config, "judge.include_reasoning"))
        and not args.exclude_reasoning
    )
    records = load_split(
        args.split,
        args.splits_dir.resolve(),
        prompt_template=str(cfg_get(training_config, "judge.prompt")),
        tokenizer=tokenizer,
        max_prompt_chars=int(cfg_get(training_config, "judge.max_prompt_chars")),
        context_truncation=str(cfg_get(training_config, "judge.context_truncation")),
        include_reasoning=include_reasoning,
        reasoning_max_chars=int(cfg_get(training_config, "judge.reasoning_max_chars")),
        reasoning_truncation=str(cfg_get(training_config, "judge.reasoning_truncation")),
        enable_thinking=bool(cfg_get(training_config, "judge.enable_thinking")),
    )
    records = apply_global_limit(records, args.global_limit)
    if args.reasoning_prefixes is not None:
        records = attach_reasoning_prefixes(records, args.reasoning_prefixes.resolve(), prefix)
    print(
        f"{args.split} rows={len(records.frame)} datasets={len(records.dataset_names)} "
        f"positives={int(records.frame['label'].sum())}",
        flush=True,
    )

    max_model_len = (
        args.max_model_len
        if args.max_model_len is not None
        else int(cfg_get(training_config, "training.max_prompt_length")) + 32
    )
    print("loading base model and adapter", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(training_config["model"]),
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.config.use_cache = True

    predictions, eval_meta = evaluate_logits(
        model=model,
        tokenizer=tokenizer,
        records=records,
        batch_size=args.batch_size,
        prefix=prefix,
        label0_id=label0_id,
        label1_id=label1_id,
        max_model_len=max_model_len,
    )
    grid = threshold_grid(predictions)
    best = grid.iloc[0].to_dict()
    baseline_metrics = macro_metrics(predictions, args.threshold)

    predictions_path = output_dir / "predictions.csv"
    grid_path = output_dir / "threshold_grid.csv"
    result_path = output_dir / "result.json"
    predictions.to_csv(predictions_path, index=False)
    grid.to_csv(grid_path, index=False)

    result = {
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": f"{training_config['method']}_logits_{args.prefix_variant}_{args.label_style}",
        "split": args.split,
        "n": int(len(predictions)),
        "metrics": baseline_metrics,
        "best_threshold": best,
        "score_summary": score_summary(predictions),
        "datasets_at_baseline": per_dataset_table(predictions, args.threshold),
        "datasets_at_best": per_dataset_table(predictions, float(best["threshold"])),
        "timing": {
            "score_seconds": eval_meta["score_time_seconds"],
            "rows_per_second": eval_meta["rows_per_second"],
            "note": "exact next-token logits; excludes model startup and dataset preparation",
        },
        "config": {
            "adapter_dir": adapter_dir.as_posix(),
            "split": args.split,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "max_model_len": max_model_len,
            "prefix_variant": args.prefix_variant,
            "label_style": args.label_style,
            "label0": label0,
            "label1": label1,
            "label0_id": label0_id,
            "label1_id": label1_id,
            "baseline_threshold": args.threshold,
            "exclude_reasoning": args.exclude_reasoning,
            "include_reasoning": include_reasoning,
            "reasoning_prefixes": args.reasoning_prefixes.as_posix() if args.reasoning_prefixes else None,
            "training_config": training_config,
        },
        "paths": {
            "predictions": predictions_path.as_posix(),
            "threshold_grid": grid_path.as_posix(),
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "metrics@threshold": baseline_metrics,
        "best_threshold": best,
        "score_summary": result["score_summary"],
        "result_path": result_path.as_posix(),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
