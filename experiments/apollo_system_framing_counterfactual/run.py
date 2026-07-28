#!/usr/bin/env python3
"""Test Apollo Justice under matched source-system framings through NDIF."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ndif_context_self_question.run import query_scores
from experiments.ndif_tvg_model_swap.run_ndif_matched_organism_tvg import (
    family_for,
    safe_group_name,
    score_stats,
    validate_group,
)
from experiments.ndif_tvg_model_swap.run_ndif_tvg_model_swap import (
    binary_metrics,
    grouped_metrics,
    load_credentials,
    per_dataset_metrics,
    prompt_sha256,
)
from experiments.phoenix_system_framing_counterfactual.run import (
    ANTI_AUDIT_RE,
    CONDITIONS,
    FRAMINGS,
    framing_messages,
    original_frame_family,
    system_content,
)
from submission.apollo_justice import (
    AUDITOR_QUESTION,
    binary_token_ids,
    has_source_reasoning,
    render_auditor_prompt,
)
from submission.util import build_model


METHOD = "apollo_system_framing_counterfactual_v1"
DEFAULT_OUTPUT_DIR = ROOT / "results/blackbox" / METHOD


def load_split_records(
    split: str,
    splits_dir: Path,
    *,
    conditions: tuple[str, ...] = CONDITIONS,
) -> list[dict[str, Any]]:
    """Load one held-out varied split and apply every matched framing."""
    from datasets import load_dataset

    manifest = yaml.safe_load(
        (splits_dir / f"dry.{split}.yaml").read_text()
    )
    records: list[dict[str, Any]] = []
    expected_source_rows = 0
    varied_declarations = [
        declaration
        for declaration in manifest["datasets"]
        if "dev-varied-deception" in str(declaration["name"])
    ]
    for declaration in varied_declarations:
        dataset_name = str(declaration["name"])
        labels_path = Path(str(declaration["labels_uri"]))
        if not labels_path.is_absolute():
            labels_path = splits_dir.parent / labels_path
        with labels_path.open(newline="") as handle:
            labels = {
                str(row[str(declaration["id_column"])]): int(
                    str(row[str(declaration["label_column"])]).casefold()
                    == "true"
                )
                for row in csv.DictReader(handle)
            }
        expected_source_rows += len(labels)
        dataset = load_dataset(dataset_name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        found: set[str] = set()
        for row in dataset:
            index = str(row["index"])
            if index not in labels:
                continue
            messages = [dict(message) for message in row["messages"]]
            original_system = system_content(messages)
            found.add(index)
            for condition in conditions:
                counterfactual_messages = framing_messages(
                    messages, condition
                )
                records.append({
                    "split": split,
                    "dataset": dataset_name,
                    "index": row["index"],
                    "label": labels[index],
                    "model": str(row["model"]),
                    "lora": row.get("lora"),
                    "condition": condition,
                    "messages": counterfactual_messages,
                    "original_system": original_system,
                    "original_frame_family": original_frame_family(
                        original_system
                    ),
                    "original_is_anti_audit": bool(
                        ANTI_AUDIT_RE.search(original_system)
                    ),
                    "source_reasoning_present": has_source_reasoning(messages),
                })
        if found != set(labels):
            raise RuntimeError(
                f"{dataset_name}: found {len(found)}/{len(labels)} labeled rows"
            )
    expected = expected_source_rows * len(conditions)
    if len(records) != expected:
        raise RuntimeError(
            f"{split}: loaded {len(records)}/{expected} counterfactual rows"
        )
    return records


def cache_keys(records: list[dict[str, Any]]) -> list[list[str]]:
    """Return condition-aware stable identifiers for cached NDIF scores."""
    return [
        [
            str(row["split"]),
            str(row["dataset"]),
            str(row["index"]),
            str(row["condition"]),
        ]
        for row in records
    ]


def validate_cached_group(
    cached: dict[str, Any],
    *,
    model_id: str,
    lora_id: str | None,
    keys: list[list[str]],
    hashes: list[str],
) -> None:
    """Reject a cache from another organism, row order, or rendered prompt."""
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
    """Query one exact generating organism in one resumable NDIF session."""
    model_id, lora_id = validate_group(records)
    group_name = safe_group_name(model_id, lora_id)
    keys = cache_keys(records)
    print(
        f"initializing group={group_name} prompts={len(records)}",
        flush=True,
    )
    model = build_model(model_id, lora_id)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        render_auditor_prompt(tokenizer, row["messages"])
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
        position_batches = [
            [int(position) for position in positions]
            for positions in cached["position_batches"]
        ]
        print(
            f"cached group={group_name} prompts={len(records)} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
    else:
        print(
            f"querying group={group_name} "
            f"p95={np.percentile(prompt_tokens, 95):.1f} "
            f"max={max(prompt_tokens)} truncation=disabled",
            flush=True,
        )
        scores, elapsed, position_batches, encoded_prompt_tokens = (
            query_scores(
                model,
                tokenizer,
                binary_token_ids(tokenizer),
                prompts,
            )
        )
        if encoded_prompt_tokens != prompt_tokens:
            raise RuntimeError(
                "prompt-length accounting changed during encoding"
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "model": model_id,
            "lora": lora_id,
            "keys": keys,
            "prompt_sha256": hashes,
            "scores": scores,
            "score_seconds": elapsed,
            "position_batches": position_batches,
            "prompt_tokens": prompt_tokens,
            "conditions": sorted(Counter(
                str(row["condition"]) for row in records
            )),
            "question": AUDITOR_QUESTION,
            "context_truncation": "none",
            "source_reasoning": "retained where supplied",
        }, indent=2) + "\n")
        print(
            f"completed group={group_name} prompts={len(records)} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

    scored_rows = []
    for row, score, prompt_hash, token_count in zip(
        records,
        scores,
        hashes,
        prompt_tokens,
        strict=True,
    ):
        scored_rows.append({
            key: value
            for key, value in row.items()
            if key not in {"messages"}
        } | {
            "family": family_for(model_id),
            "score": float(score),
            "prompt_sha256": prompt_hash,
            "prompt_tokens": int(token_count),
        })
    metadata = {
        "model": model_id,
        "lora": lora_id,
        "prompts": len(records),
        "score_seconds": elapsed,
        "position_batches": position_batches,
        "prompt_tokens": prompt_tokens,
        "conditions": dict(Counter(
            str(row["condition"]) for row in records
        )),
    }
    return scored_rows, metadata


def paired_comparison(
    rows: list[dict[str, Any]],
    condition: str,
    *,
    baseline: str = "generic_deception",
) -> dict[str, Any]:
    """Compare paired score shifts and per-dataset AUROC changes."""
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(
        index=["split", "dataset", "index", "label"],
        columns="condition",
        values="score",
    ).dropna(subset=[baseline, condition])
    delta = pivot[condition] - pivot[baseline]
    labels = pivot.index.get_level_values("label")
    correlations = (
        pivot[[baseline, condition]].corr() if len(pivot) > 1 else None
    )
    rank_correlations = (
        pivot[[baseline, condition]].corr(method="spearman")
        if len(pivot) > 1 else None
    )

    unit_deltas = []
    for dataset, group in frame.groupby("dataset"):
        condition_rows = group[group["condition"] == condition]
        baseline_rows = group[group["condition"] == baseline]
        condition_auroc = binary_metrics(
            condition_rows["label"].to_numpy(dtype=int),
            condition_rows["score"].to_numpy(dtype=float),
        )["auroc"]
        baseline_auroc = binary_metrics(
            baseline_rows["label"].to_numpy(dtype=int),
            baseline_rows["score"].to_numpy(dtype=float),
        )["auroc"]
        delta_value = (
            None
            if condition_auroc is None or baseline_auroc is None
            else float(condition_auroc - baseline_auroc)
        )
        unit_deltas.append({
            "dataset": dataset,
            "delta": delta_value,
        })
    defined_deltas = [
        row["delta"] for row in unit_deltas if row["delta"] is not None
    ]
    honest_delta = delta[labels == 0]
    deceptive_delta = delta[labels == 1]
    return {
        "paired_rows": int(len(pivot)),
        "mean_score_delta": float(delta.mean()),
        "mean_score_delta_honest": (
            float(honest_delta.mean()) if len(honest_delta) else None
        ),
        "mean_score_delta_deceptive": (
            float(deceptive_delta.mean()) if len(deceptive_delta) else None
        ),
        "median_score_delta": float(delta.median()),
        "spearman": (
            float(rank_correlations.iloc[0, 1])
            if rank_correlations is not None else None
        ),
        "pearson": (
            float(correlations.iloc[0, 1])
            if correlations is not None else None
        ),
        "dataset_wins": int(sum(value > 0 for value in defined_deltas)),
        "dataset_ties": int(sum(value == 0 for value in defined_deltas)),
        "dataset_losses": int(sum(value < 0 for value in defined_deltas)),
        "dataset_auroc_undefined": int(
            len(unit_deltas) - len(defined_deltas)
        ),
        "per_dataset_auroc_delta": unit_deltas,
    }


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one split under all five matched system conditions."""
    report: dict[str, Any] = {"conditions": {}, "comparisons": {}}
    for condition in CONDITIONS:
        values = [
            row for row in rows if row["condition"] == condition
        ]
        labels = np.asarray([row["label"] for row in values], dtype=int)
        scores = np.asarray([row["score"] for row in values], dtype=float)
        report["conditions"][condition] = {
            "rows": len(values),
            "metrics": grouped_metrics(values),
            "pooled_metrics": binary_metrics(labels, scores),
            "score_stats": score_stats(values),
            "mean_score_honest": (
                float(scores[labels == 0].mean())
                if np.any(labels == 0) else None
            ),
            "mean_score_deceptive": (
                float(scores[labels == 1].mean())
                if np.any(labels == 1) else None
            ),
            "per_dataset": per_dataset_metrics(values),
        }
    for condition in CONDITIONS:
        if condition == "generic_deception":
            continue
        report["comparisons"][
            f"{condition}_vs_generic_deception"
        ] = paired_comparison(rows, condition)

    original = [
        row for row in rows if row["condition"] == "original"
    ]
    report["original_frame_counts"] = dict(sorted(Counter(
        str(row["original_frame_family"]) for row in original
    ).items()))
    anti_audit_keys = {
        (str(row["dataset"]), str(row["index"]))
        for row in original
        if row["original_is_anti_audit"]
    }
    report["original_anti_audit_rows"] = len(anti_audit_keys)
    report["anti_audit_pooled"] = {}
    for condition in CONDITIONS:
        values = [
            row for row in rows
            if row["condition"] == condition
            and (str(row["dataset"]), str(row["index"])) in anti_audit_keys
        ]
        report["anti_audit_pooled"][condition] = binary_metrics(
            np.asarray([row["label"] for row in values], dtype=int),
            np.asarray([row["score"] for row in values], dtype=float),
        )
    return report


def summarize(
    rows: list[dict[str, Any]],
    group_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the frozen AUROC-first experiment report."""
    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    score_seconds = float(sum(
        metadata["score_seconds"]
        for metadata in group_metadata.values()
    ))
    return {
        "method": METHOD,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(rows),
        "conditions": list(CONDITIONS),
        "organisms": len(group_metadata),
        "question": AUDITOR_QUESTION,
        "prompt_contract": {
            "exact_generating_organism": True,
            "source_reasoning": "retained where supplied",
            "counterfactual_reasoning_policy": (
                "original native trace held fixed when system text changes"
            ),
            "context_truncation": "none",
            "enable_thinking": False,
            "prefix": "Prediction:",
            "score": "softmax(logit_0, logit_1)[1]",
            "remote_backend": "competition NDIF",
        },
        "framing_text": FRAMINGS,
        "splits": {
            split: condition_summary([
                row for row in rows if row["split"] == split
            ])
            for split in ("validation", "test")
        },
        "prompt_tokens": {
            "minimum": int(np.min(prompt_tokens)),
            "median": float(np.median(prompt_tokens)),
            "p95": float(np.percentile(prompt_tokens, 95)),
            "maximum": int(np.max(prompt_tokens)),
            "truncated": 0,
        },
        "timing": {
            "score_seconds": score_seconds,
            "rows_per_second": len(rows) / score_seconds,
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
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=ROOT / "dev_splits"
    )
    parser.add_argument(
        "--group-regex",
        help="score only exact-organism safe names matching this regex",
    )
    parser.add_argument(
        "--limit-per-group",
        type=int,
        help="limit source rows per split and organism before fivefold expansion",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit_per_group is not None and args.limit_per_group < 1:
        parser.error("--limit-per-group must be positive")
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    load_credentials()
    source_records = {
        split: load_split_records(
            split,
            args.splits_dir.resolve(),
            conditions=("original",),
        )
        for split in ("validation", "test")
    }
    # Re-expand after optional source-row limits so every retained row has all
    # five conditions.
    selected_source: list[dict[str, Any]] = []
    grouped_source: dict[
        tuple[str, str | None, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for split, records in source_records.items():
        for row in records:
            group = (
                str(row["model"]),
                None if row.get("lora") in (None, "") else str(row["lora"]),
                split,
            )
            grouped_source[group].append(row)
    for values in grouped_source.values():
        selected_source.extend(
            values[:args.limit_per_group]
            if args.limit_per_group is not None
            else values
        )

    records: list[dict[str, Any]] = []
    for row in selected_source:
        original_messages = row["messages"]
        original_system = row["original_system"]
        for condition in CONDITIONS:
            expanded = dict(row)
            expanded["condition"] = condition
            expanded["messages"] = framing_messages(
                original_messages, condition
            )
            expanded["original_system"] = original_system
            records.append(expanded)

    organism_groups: dict[
        tuple[str, str | None], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in records:
        key = (
            str(row["model"]),
            None if row.get("lora") in (None, "") else str(row["lora"]),
        )
        organism_groups[key].append(row)
    if args.group_regex:
        pattern = re.compile(args.group_regex)
        organism_groups = {
            key: values
            for key, values in organism_groups.items()
            if pattern.search(safe_group_name(*key))
        }
    if not organism_groups:
        raise RuntimeError("selected exact-organism groups are empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output_dir / "groups"
    all_rows: list[dict[str, Any]] = []
    group_metadata: dict[str, dict[str, Any]] = {}
    for (model_id, lora_id), group in sorted(
        organism_groups.items(), key=lambda item: str(item[0])
    ):
        group_name = safe_group_name(model_id, lora_id)
        scored, metadata = query_group(
            group,
            groups_dir / f"{group_name}.json",
            overwrite=args.overwrite,
        )
        all_rows.extend(scored)
        group_metadata[group_name] = metadata

    split_order = {"validation": 0, "test": 1}
    condition_order = {
        condition: position
        for position, condition in enumerate(CONDITIONS)
    }
    all_rows.sort(key=lambda row: (
        split_order[str(row["split"])],
        str(row["dataset"]),
        str(row["index"]),
        condition_order[str(row["condition"])],
    ))
    (args.output_dir / "scores.jsonl").write_text(
        "\n".join(json.dumps(row) for row in all_rows) + "\n"
    )
    result = summarize(all_rows, group_metadata)
    result["wall_seconds"] = time.time() - started
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "method": result["method"],
        "rows": result["rows"],
        "organisms": result["organisms"],
        "splits": {
            split: {
                condition: values["metrics"]["auroc"]
                for condition, values in report["conditions"].items()
            }
            for split, report in result["splits"].items()
        },
        "prompt_tokens": result["prompt_tokens"],
        "timing": result["timing"],
        "wall_seconds": result["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
