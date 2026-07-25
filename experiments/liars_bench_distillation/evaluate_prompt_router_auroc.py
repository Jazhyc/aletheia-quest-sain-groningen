#!/usr/bin/env python3
"""Evaluate direct binary-token margins for the frozen semantic prompt router."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.compose_prompt_router import route_matches
from experiments.liars_bench_distillation.evaluate_prompt_router import render
from experiments.liars_bench_distillation.evaluate_prompt_sweep import (
    load_prompt,
    split_records,
)
from experiments.privileged_information_distillation.evaluate_student_sft import (
    DIRECT_PREDICTION_PREFIX,
    binary_token_ids,
    score_binary_prefixes,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import binary_metrics


SCORE_COMPOSITIONS = ("control", "logit_mean", "specialist")


def logit(score: float, *, epsilon: float = 1e-12) -> float:
    """Convert a probability to finite log odds."""
    clipped = min(1.0 - epsilon, max(epsilon, float(score)))
    return math.log(clipped) - math.log1p(-clipped)


def sigmoid(value: float) -> float:
    """Convert finite log odds to a probability."""
    value = max(-80.0, min(80.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def compose_score(
    control_score: float,
    specialist_score: float | None,
    *,
    composition: str,
) -> float:
    """Compose general and routed specialist margins on a common score scale."""
    if composition == "control" or specialist_score is None:
        return float(control_score)
    if composition == "specialist":
        return float(specialist_score)
    if composition == "logit_mean":
        return sigmoid((logit(control_score) + logit(specialist_score)) / 2.0)
    raise ValueError(f"unsupported score composition: {composition!r}")


def compose_margin_rows(
    records: list[dict[str, Any]],
    control_scores: list[float],
    specialist_scores: dict[tuple[str, str], float],
    *,
    route_kind: str,
    composition: str,
) -> list[dict[str, Any]]:
    """Attach an AUROC score while preserving the exact label-blind route."""
    if len(records) != len(control_scores):
        raise ValueError(
            f"received {len(control_scores)} control scores for {len(records)} rows"
        )
    rows = []
    for record, control_score in zip(records, control_scores, strict=True):
        key = (str(record["dataset"]), str(record["index"]))
        routed = route_matches(record["messages"], route_kind)
        specialist_score = specialist_scores.get(key)
        if routed != (specialist_score is not None):
            raise ValueError(
                f"specialist score coverage disagrees with {route_kind} route for {key}"
            )
        rows.append({
            **{
                field: record[field]
                for field in (
                    "dataset",
                    "index",
                    "category",
                    "source_model",
                    "label",
                )
            },
            "score": compose_score(
                control_score,
                specialist_score,
                composition=composition,
            ),
            "control_score": float(control_score),
            "specialist_score": (
                None if specialist_score is None else float(specialist_score)
            ),
            "routed": routed,
            "route_kind": route_kind if routed else None,
            "score_composition": composition,
        })
    return rows


def grouped_metrics(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    """Compute threshold diagnostics and AUROC for observable report groups."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = "::".join(str(row[field]) for field in fields)
        groups[name].append(row)
    return {
        name: binary_metrics(
            np.asarray([row["label"] for row in group], dtype=np.int64),
            np.asarray([row["score"] for row in group], dtype=np.float64),
            0.5,
        )
        for name, group in sorted(groups.items())
    }


def summarize(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """Summarize AUROC-first metrics while retaining threshold diagnostics."""
    per_dataset = grouped_metrics(rows, ("dataset",))
    aurocs = [
        metrics["auroc"]
        for metrics in per_dataset.values()
        if metrics["auroc"] is not None
    ]
    return {
        "rows": len(rows),
        "macro_dataset_auroc": float(np.mean(aurocs)) if aurocs else None,
        "metrics": binary_metrics(
            np.asarray([row["label"] for row in rows], dtype=np.int64),
            np.asarray([row["score"] for row in rows], dtype=np.float64),
            0.5,
        ),
        "per_dataset": per_dataset,
        "per_category": grouped_metrics(rows, ("category",)),
        "per_source_model": grouped_metrics(rows, ("source_model",)),
        "per_category_source_model": grouped_metrics(
            rows, ("category", "source_model")
        ),
        "unique_scores": len({float(row["score"]) for row in rows}),
        "score_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed > 0 else None,
    }


def select_composition(
    summaries: dict[str, dict[str, Any]],
    *,
    minimum_macro_gain: float,
) -> dict[str, Any]:
    """Select one development composition, preferring simpler ties."""
    baseline = float(summaries["control"]["macro_dataset_auroc"])
    comparisons = {}
    for name in SCORE_COMPOSITIONS:
        score = float(summaries[name]["macro_dataset_auroc"])
        comparisons[name] = {
            "macro_dataset_auroc": score,
            "gain": score - baseline,
            "passes": name == "control" or score - baseline >= minimum_macro_gain,
        }
    eligible = [
        name for name in SCORE_COMPOSITIONS
        if comparisons[name]["passes"]
    ]
    selected = max(
        eligible,
        key=lambda name: (
            comparisons[name]["macro_dataset_auroc"],
            -SCORE_COMPOSITIONS.index(name),
        ),
    )
    return {
        "metric": "macro_dataset_auroc",
        "minimum_macro_gain": minimum_macro_gain,
        "selected": selected,
        "comparisons": comparisons,
    }


def load_selected(path: Path) -> str:
    selected = str(json.loads(path.read_text())["selected"])
    if selected not in SCORE_COMPOSITIONS:
        raise ValueError(f"invalid selected composition: {selected!r}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--specialist-config", type=Path, required=True)
    parser.add_argument(
        "--route-kind",
        choices=("knowledge", "choice", "union"),
        default="knowledge",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "confirmation"),
        required=True,
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260725)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--minimum-macro-gain", type=float, default=0.005)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument(
        "--context-truncation",
        choices=("head_tail", "tail"),
        default="head_tail",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.split == "confirmation" and args.selection is None:
        raise ValueError("--selection is required for the confirmation split")
    if args.split == "development" and args.selection is not None:
        raise ValueError("--selection is only valid for the confirmation split")

    all_records = [
        json.loads(line)
        for line in args.eval_artifact.read_text().splitlines()
        if line.strip()
    ]
    records = split_records(all_records, args.split, seed=args.split_seed)
    if args.expected_rows is not None and len(records) != args.expected_rows:
        raise ValueError(
            f"{args.split} contains {len(records)} rows, expected {args.expected_rows}"
        )
    routed_records = [
        row for row in records if route_matches(row["messages"], args.route_kind)
    ]
    coverage = {
        "rows": len(routed_records),
        "per_category": dict(sorted(Counter(
            str(row["category"]) for row in routed_records
        ).items())),
        "per_source_model": dict(sorted(Counter(
            str(row["source_model"]) for row in routed_records
        ).items())),
    }

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_dir = args.adapter_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    llm = LLM(
        model=args.model,
        tokenizer=adapter_dir.as_posix(),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=16,
    )
    request = LoRARequest("liars-prompt-router-auroc", 1, adapter_dir.as_posix())
    binary_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(binary_ids),
        logprob_token_ids=binary_ids,
        allowed_token_ids=binary_ids,
    )

    control_prompts = [
        prompt + DIRECT_PREDICTION_PREFIX
        for prompt in render(
            tokenizer,
            records,
            load_prompt(args.control_config.resolve()),
            max_prompt_chars=args.max_prompt_chars,
            context_truncation=args.context_truncation,
        )
    ]
    control_scores, control_missing, control_elapsed = score_binary_prefixes(
        llm, control_prompts, sampling, request, binary_ids
    )

    specialist_prompts = [
        prompt + DIRECT_PREDICTION_PREFIX
        for prompt in render(
            tokenizer,
            routed_records,
            load_prompt(args.specialist_config.resolve()),
            max_prompt_chars=args.max_prompt_chars,
            context_truncation=args.context_truncation,
        )
    ]
    specialist_values, specialist_missing, specialist_elapsed = score_binary_prefixes(
        llm, specialist_prompts, sampling, request, binary_ids
    )
    specialist_scores = {
        (str(row["dataset"]), str(row["index"])): score
        for row, score in zip(routed_records, specialist_values, strict=True)
    }
    prompt_hashes = {
        "control": hashlib.sha256(
            load_prompt(args.control_config.resolve()).encode("utf-8")
        ).hexdigest(),
        "specialist": hashlib.sha256(
            load_prompt(args.specialist_config.resolve()).encode("utf-8")
        ).hexdigest(),
    }

    if args.split == "development":
        compositions = SCORE_COMPOSITIONS
    else:
        selected = load_selected(args.selection.resolve())
        compositions = ("control",) if selected == "control" else ("control", selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    elapsed = control_elapsed + specialist_elapsed
    for composition in compositions:
        rows = compose_margin_rows(
            records,
            control_scores,
            specialist_scores,
            route_kind=args.route_kind,
            composition=composition,
        )
        summaries[composition] = summarize(rows, elapsed)
        with (args.output_dir / f"{composition}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result: dict[str, Any] = {
        "split": args.split,
        "split_seed": args.split_seed,
        "rows": len(records),
        "metric": "macro_dataset_auroc",
        "score_method": "direct constrained Prediction: 0/1 margin",
        "route_kind": args.route_kind,
        "route_coverage": coverage,
        "prompt_sha256": prompt_hashes,
        "margin_missing": {
            "control": control_missing,
            "specialist": specialist_missing,
        },
        "timing": {
            "control_margin_seconds": control_elapsed,
            "specialist_margin_seconds": specialist_elapsed,
            "total_margin_seconds": elapsed,
        },
        "conditions": summaries,
    }
    if args.split == "development":
        selection = select_composition(
            summaries,
            minimum_macro_gain=args.minimum_macro_gain,
        )
        result["selection"] = selection
        (args.output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
    else:
        result["frozen_selection"] = load_selected(args.selection.resolve())
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
