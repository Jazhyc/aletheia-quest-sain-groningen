#!/usr/bin/env python3
"""Freeze an untouched passage-grounded True/False confirmation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.liars_bench_distillation.prepare_teacher_data import (
    load_category,
    stable_sample,
    write_jsonl,
)
from experiments.liars_bench_distillation.router import is_passage_true_false_request


def ids_from_artifact(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids = set()
    for row in map(json.loads, path.read_text().splitlines()):
        category = row.get("category") or str(row.get("dataset", "")).split("/")[-1]
        if category == "soft-trigger":
            ids.add(str(row["index"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--liars-root", type=Path)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    excluded = set().union(*(ids_from_artifact(path) for path in args.exclude))
    frame = load_category("soft-trigger", args.liars_root)
    frame = frame[~frame["sample_id"].isin(excluded)].reset_index(drop=True)
    selected = stable_sample(frame, per_label=args.per_label, seed=args.seed)
    records = [
        {
            "dataset": "liars-bench/soft-trigger",
            "index": row["sample_id"],
            "category": "soft-trigger",
            "source_model": row["source_model"],
            "label": int(row["label"]),
            "messages": row["messages"],
        }
        for row in selected.to_dict(orient="records")
    ]
    if not all(is_passage_true_false_request(row["messages"]) for row in records):
        raise RuntimeError("selected row violates the frozen passage router")
    write_jsonl(args.output, records)
    print(json.dumps({
        "rows": len(records),
        "excluded": len(excluded),
        "seed": args.seed,
        "source_counts": selected["source_model"].value_counts().sort_index().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
