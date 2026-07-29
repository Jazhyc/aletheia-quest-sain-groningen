#!/usr/bin/env python3
"""Score each frozen test row with its exact generating organism over NDIF."""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ndif_tvg_model_swap.run_ndif_tvg_model_swap import (
    MAX_PROMPT_CHARS,
    MAX_PROMPT_TOKENS,
    PREDICTION_PREFIX,
    TRUTH_VALUE_GUARD_PROMPT,
    binary_token_ids,
    grouped_metrics,
    load_credentials,
    load_test_records,
    per_dataset_metrics,
    prompt_sha256,
    query_score_shard,
    render_chat_prompt,
)
from submission.util import build_model


METHOD = "ndif_matched_organism_tvg_test_v1"


def batch_size_for(model_id: str) -> int:
    """Use NDIF batch sizes already validated for each hosted family."""
    folded = model_id.casefold()
    if "nemotron" in folded:
        return 1
    if "gemma" in folded:
        return 8
    if "qwen" in folded:
        return 16
    return 4


def family_for(model_id: str) -> str:
    folded = model_id.casefold()
    if "nemotron" in folded:
        return "nemotron"
    if "gemma" in folded:
        return "gemma"
    if "qwen" in folded:
        return "qwen"
    return model_id


def safe_group_name(model_id: str, lora_id: str | None) -> str:
    value = f"{model_id}__{lora_id or 'base'}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def score_stats(
    rows: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    if not rows:
        return {
            "unique": 0,
            "ties": 0,
            "minimum": None,
            "mean": None,
            "maximum": None,
            "exact_zero": 0,
            "exact_one": 0,
        }
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    unique = int(np.unique(scores).size)
    return {
        "unique": unique,
        "ties": int(len(scores) - unique),
        "minimum": float(scores.min()),
        "mean": float(scores.mean()),
        "maximum": float(scores.max()),
        "exact_zero": int(np.count_nonzero(scores == 0.0)),
        "exact_one": int(np.count_nonzero(scores == 1.0)),
    }


def breakdown(
    rows: list[dict[str, Any]],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return {
        name: {
            "rows": len(values),
            "dataset_units": len({row["dataset"] for row in values}),
            "metrics": grouped_metrics(values),
            "score_stats": score_stats(values),
        }
        for name, values in sorted(groups.items())
    }


def validate_group(records: list[dict[str, Any]]) -> tuple[str, str | None]:
    organisms = {
        (
            str(row["model"]),
            None if row.get("lora") in (None, "") else str(row["lora"]),
        )
        for row in records
    }
    if len(organisms) != 1:
        raise ValueError(
            f"group contains multiple organisms: "
            f"{sorted(organisms, key=str)}"
        )
    return next(iter(organisms))


def validate_cached_group(
    cached: dict[str, Any],
    *,
    model_id: str,
    lora_id: str | None,
    keys: list[list[str]],
    hashes: list[str],
) -> None:
    if cached.get("model") != model_id:
        raise ValueError("cached group model mismatch")
    if cached.get("lora") != lora_id:
        raise ValueError("cached group LoRA mismatch")
    if cached.get("keys") != keys:
        raise ValueError("cached group row-key mismatch")
    if cached.get("prompt_sha256") != hashes:
        raise ValueError("cached group prompt mismatch")
    if len(cached.get("scores", [])) != len(keys):
        raise ValueError("cached group score count mismatch")


def query_group(
    records: list[dict[str, Any]],
    cache_path: Path,
    *,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query or restore one exact base-model/LoRA organism."""
    model_id, lora_id = validate_group(records)
    group_name = safe_group_name(model_id, lora_id)
    batch_size = batch_size_for(model_id)
    keys = [
        [str(row["dataset"]), str(row["index"])]
        for row in records
    ]

    print(
        f"initializing group={group_name} rows={len(records)} "
        f"batch={batch_size}",
        flush=True,
    )
    model = build_model(model_id, lora_id)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = binary_token_ids(tokenizer)
    prompts = [
        render_chat_prompt(tokenizer, row["messages"])
        for row in records
    ]
    hashes = [prompt_sha256(prompt) for prompt in prompts]
    prompt_tokens = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]

    if cache_path.exists() and not overwrite:
        cached = json.loads(cache_path.read_text())
        validate_cached_group(
            cached,
            model_id=model_id,
            lora_id=lora_id,
            keys=keys,
            hashes=hashes,
        )
        scores = [float(value) for value in cached["scores"]]
        elapsed = float(cached["score_seconds"])
        print(
            f"cached group={group_name} rows={len(records)} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
    else:
        print(
            f"querying group={group_name} label_ids={label_ids} "
            f"prompt_tokens_p95={np.percentile(prompt_tokens, 95):.1f}",
            flush=True,
        )
        scores, elapsed = query_score_shard(
            model,
            tokenizer,
            label_ids,
            prompts,
            batch_size,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "model": model_id,
            "lora": lora_id,
            "keys": keys,
            "prompt_sha256": hashes,
            "scores": scores,
            "score_seconds": elapsed,
            "batch_size": batch_size,
            "label_token_ids": label_ids,
            "prompt_tokens": prompt_tokens,
        }, indent=2) + "\n")
        print(
            f"completed group={group_name} rows={len(records)} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

    rows = [
        {
            "dataset": row["dataset"],
            "index": row["index"],
            "label": row["label"],
            "model": model_id,
            "lora": lora_id,
            "family": family_for(model_id),
            "score": score,
            "prompt_sha256": prompt_hash,
        }
        for row, score, prompt_hash in zip(
            records,
            scores,
            hashes,
            strict=True,
        )
    ]
    metadata = {
        "model": model_id,
        "lora": lora_id,
        "rows": len(rows),
        "batch_size": batch_size,
        "label_token_ids": label_ids,
        "score_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
    }
    return rows, metadata


def summarize(
    rows: list[dict[str, Any]],
    group_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scenario_metrics = {}
    for scenario in ("instructed-deception", "varied-deception"):
        values = [row for row in rows if scenario in str(row["dataset"])]
        scenario_metrics[scenario] = {
            "rows": len(values),
            "metrics": grouped_metrics(values),
            "score_stats": score_stats(values),
        }

    dataset_details = []
    metrics_by_dataset = {
        value["dataset"]: value
        for value in per_dataset_metrics(rows)
    }
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    for dataset, values in sorted(by_dataset.items()):
        model_id, lora_id = validate_group(values)
        dataset_details.append({
            **metrics_by_dataset[dataset],
            "model": model_id,
            "lora": lora_id,
            "family": family_for(model_id),
            "score_stats": score_stats(values),
        })

    all_prompt_tokens = [
        token_count
        for metadata in group_metadata.values()
        for token_count in metadata["prompt_tokens"]
    ]
    total_seconds = float(sum(
        metadata["score_seconds"]
        for metadata in group_metadata.values()
    ))
    return {
        "method": METHOD,
        "split": "test",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(rows),
        "dataset_units": len(by_dataset),
        "organisms": len(group_metadata),
        "prompt": TRUTH_VALUE_GUARD_PROMPT,
        "prompt_contract": {
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "context_truncation": "tail",
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "enable_thinking": False,
            "prefix": PREDICTION_PREFIX,
            "label_texts": ["0", "1"],
            "score": "softmax(logit_0, logit_1)[1]",
            "remote_backend": "competition NDIF",
        },
        "metrics": grouped_metrics(rows),
        "scenarios": scenario_metrics,
        "families": breakdown(rows, lambda row: row["family"]),
        "adapter_status": breakdown(
            rows,
            lambda row: "base" if row["lora"] is None else "lora",
        ),
        "per_organism": breakdown(
            rows,
            lambda row: f"{row['model']}::{row['lora'] or 'base'}",
        ),
        "per_dataset": dataset_details,
        "score_stats": score_stats(rows),
        "prompt_tokens": {
            "minimum": int(np.min(all_prompt_tokens)),
            "median": float(np.median(all_prompt_tokens)),
            "p95": float(np.percentile(all_prompt_tokens, 95)),
            "maximum": int(np.max(all_prompt_tokens)),
        },
        "timing": {
            "score_seconds": total_seconds,
            "rows_per_second": len(rows) / total_seconds,
            "per_organism": {
                name: metadata["score_seconds"]
                for name, metadata in group_metadata.items()
            },
        },
        "group_metadata": group_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/blackbox" / METHOD,
    )
    parser.add_argument(
        "--dataset-regex",
        help="only score dataset names matching this regular expression",
    )
    parser.add_argument(
        "--limit-per-organism",
        type=int,
        help="truncate every exact organism group (for compatibility smokes)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit_per_organism is not None and args.limit_per_organism < 1:
        parser.error("--limit-per-organism must be positive")
    return args


def main() -> None:
    args = parse_args()
    load_credentials()
    records = load_test_records()
    if args.dataset_regex:
        pattern = re.compile(args.dataset_regex)
        records = [
            row for row in records
            if pattern.search(str(row["dataset"]))
        ]
    if not records:
        raise RuntimeError("selected test rows are empty")

    organism_groups: dict[tuple[str, str | None], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in records:
        key = (
            str(row["model"]),
            None if row.get("lora") in (None, "") else str(row["lora"]),
        )
        organism_groups[key].append(row)
    if args.limit_per_organism is not None:
        organism_groups = {
            key: values[:args.limit_per_organism]
            for key, values in organism_groups.items()
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output_dir / "groups"
    all_rows = []
    group_metadata = {}
    for (model_id, lora_id), group in sorted(
        organism_groups.items(),
        key=lambda item: str(item[0]),
    ):
        group_name = safe_group_name(model_id, lora_id)
        rows, metadata = query_group(
            group,
            groups_dir / f"{group_name}.json",
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        group_metadata[group_name] = metadata

    order = {
        (str(row["dataset"]), str(row["index"])): position
        for position, row in enumerate(records)
    }
    all_rows.sort(
        key=lambda row: order[(str(row["dataset"]), str(row["index"]))]
    )
    (args.output_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in all_rows) + "\n"
    )
    result = summarize(all_rows, group_metadata)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "method": METHOD,
        "rows": len(all_rows),
        "dataset_units": result["dataset_units"],
        "organisms": result["organisms"],
        "metrics": result["metrics"],
        "scenarios": result["scenarios"],
        "families": result["families"],
        "adapter_status": result["adapter_status"],
        "score_stats": result["score_stats"],
        "timing": result["timing"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
