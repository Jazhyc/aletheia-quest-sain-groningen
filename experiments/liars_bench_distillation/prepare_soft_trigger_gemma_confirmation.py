#!/usr/bin/env python3
"""Freeze an untouched Gemma soft-trigger confirmation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.liars_bench_distillation.prepare_teacher_data import (
    load_category,
    stable_sample,
    write_jsonl,
)


def used_ids(train_artifact: Path, eval_artifact: Path) -> set[str]:
    used = {
        row["index"]
        for row in map(json.loads, eval_artifact.read_text().splitlines())
        if row["category"] == "soft-trigger"
    }
    used.update(
        row["index"]
        for row in map(json.loads, train_artifact.read_text().splitlines())
        if row["dataset"] == "liars-bench/soft-trigger"
    )
    return used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--liars-root", type=Path)
    parser.add_argument("--train-artifact", type=Path, required=True)
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    frame = load_category("soft-trigger", args.liars_root)
    frame = frame[
        frame["source_model"].str.lower().str.contains("gemma")
        & ~frame["sample_id"].isin(used_ids(args.train_artifact, args.eval_artifact))
    ].reset_index(drop=True)
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
    if len(records) != 2 * args.per_label:
        raise RuntimeError("failed to produce the requested balanced confirmation set")
    write_jsonl(args.output, records)
    print(json.dumps({
        "rows": len(records),
        "per_label": args.per_label,
        "seed": args.seed,
        "source_counts": selected["source_model"].value_counts().sort_index().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
