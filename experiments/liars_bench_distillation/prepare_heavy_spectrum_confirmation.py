#!/usr/bin/env python3
"""Freeze an untouched balanced confirmation set for the Q9/Q27 heavy swap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.liars_bench_distillation.prepare_teacher_data import (
    DEFAULT_CATEGORIES,
    load_category,
    stable_sample,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXCLUSIONS = (
    ROOT / "results/blackbox/liars_bench_pid_aug_v1/eval.jsonl",
    ROOT / "results/blackbox/liars_bench_pid_aug_v1/teacher/train.jsonl",
    ROOT / "results/blackbox/liars_bench_soft_trigger_gemma_confirmation_v1/eval.jsonl",
    ROOT / "results/blackbox/liars_bench_passage_true_false_v1/eval.jsonl",
)


def exclusion_ids(paths: list[Path]) -> set[str]:
    excluded = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            index = str(row["index"])
            category = str(row.get("category") or str(row["dataset"]).split("/")[-1])
            excluded.add(index if index.startswith(f"{category}:") else f"{category}:{index}")
    return excluded


def make_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "dataset": f"liars-bench/{row['category']}",
            "index": str(row["sample_id"]),
            "category": str(row["category"]),
            "source_model": str(row["source_model"]),
            "label": int(row["label"]),
            "messages": row["messages"],
        }
        for row in frame.to_dict(orient="records")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--liars-root", type=Path)
    parser.add_argument("--per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--exclude", type=Path, action="append")
    args = parser.parse_args()

    exclusions = list(DEFAULT_EXCLUSIONS if args.exclude is None else args.exclude)
    excluded = exclusion_ids(exclusions)
    parts = []
    for offset, category in enumerate(DEFAULT_CATEGORIES):
        frame = load_category(category, args.liars_root)
        selected = stable_sample(
            frame,
            per_label=args.per_label,
            seed=args.seed + offset,
            excluded_ids=excluded,
        )
        if set(selected["sample_id"]) & excluded:
            raise RuntimeError(f"exclusion leakage in {category}")
        parts.append(selected)
        print(
            f"category={category} rows={len(selected)} "
            f"models={dict(selected['source_model'].value_counts())}",
            flush=True,
        )
    records = make_records(pd.concat(parts, ignore_index=True))
    expected = len(DEFAULT_CATEGORIES) * args.per_label * 2
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(records)}")
    write_jsonl(args.output, records)
    print(f"wrote {args.output} rows={len(records)} excluded_ids={len(excluded)}")


if __name__ == "__main__":
    main()
