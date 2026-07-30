#!/usr/bin/env python3
"""Build a fixed dataset/label-stratified Kimi scale-selection holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load unique labeled student rows."""
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    keys = [(str(row["dataset"]), row["index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("student rows contain duplicate dataset/index keys")
    return rows


def stable_train_keys(
    rows: list[dict[str, Any]],
    fraction: float,
    seed: int,
) -> set[tuple[str, Any]]:
    """Select a stable fraction within every dataset/label stratum."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("train fraction must be strictly between zero and one")
    strata: dict[
        tuple[str, int],
        list[tuple[bytes, tuple[str, Any]]],
    ] = defaultdict(list)
    for row in rows:
        dataset = str(row["dataset"])
        label = int(row["label"])
        index = row["index"]
        digest = hashlib.sha256(
            f"{seed}\0{dataset}\0{label}\0{index}".encode("utf-8")
        ).digest()
        strata[(dataset, label)].append((digest, (dataset, index)))

    selected: set[tuple[str, Any]] = set()
    for stratum, candidates in strata.items():
        count = max(1, int(len(candidates) * fraction + 0.5))
        if count >= len(candidates):
            raise ValueError(f"holdout is empty for stratum={stratum!r}")
        selected.update(key for _, key in sorted(candidates)[:count])
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def build(
    student_rows: Path,
    output_dir: Path,
    *,
    train_fraction: float,
    seed: int,
    expected_rows: int,
) -> dict[str, Any]:
    """Write an explicit train manifest and evaluator-compatible holdout split."""
    rows = load_rows(student_rows)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")
    train_keys = stable_train_keys(rows, train_fraction, seed)
    train_rows = [
        {
            "dataset": str(row["dataset"]),
            "index": row["index"],
            "label": int(row["label"]),
        }
        for row in rows
        if (str(row["dataset"]), row["index"]) in train_keys
    ]
    holdout_rows = [
        row
        for row in rows
        if (str(row["dataset"]), row["index"]) not in train_keys
    ]
    write_jsonl(output_dir / "train_manifest.jsonl", train_rows)

    labels_dir = output_dir / "holdout_splits" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in holdout_rows:
        by_dataset[str(row["dataset"])].append(row)
    datasets = []
    for dataset, group in sorted(by_dataset.items()):
        if {int(row["label"]) for row in group} != {0, 1}:
            raise ValueError(f"holdout dataset lacks both labels: {dataset}")
        label_path = labels_dir / (dataset.replace("/", "__") + ".csv")
        with label_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "deceptive"])
            writer.writeheader()
            for row in sorted(group, key=lambda item: str(item["index"])):
                writer.writerow({
                    "index": row["index"],
                    "deceptive": bool(row["label"]),
                })
        datasets.append({
            "name": dataset,
            "labels_uri": label_path.resolve().as_posix(),
            "id_column": "index",
            "label_column": "deceptive",
        })
    split_path = output_dir / "holdout_splits" / "dry.train.yaml"
    split_path.write_text(yaml.safe_dump({"datasets": datasets}, sort_keys=False))

    train_counts = Counter(
        (str(row["dataset"]), int(row["label"])) for row in train_rows
    )
    holdout_counts = Counter(
        (str(row["dataset"]), int(row["label"])) for row in holdout_rows
    )
    audit = {
        "source": student_rows.resolve().as_posix(),
        "source_sha256": hashlib.sha256(student_rows.read_bytes()).hexdigest(),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "datasets": len(by_dataset),
        "train_fraction": train_fraction,
        "seed": seed,
        "train_counts": {
            f"{dataset}|{label}": count
            for (dataset, label), count in sorted(train_counts.items())
        },
        "holdout_counts": {
            f"{dataset}|{label}": count
            for (dataset, label), count in sorted(holdout_counts.items())
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--expected-rows", type=int, default=6573)
    args = parser.parse_args()
    audit = build(
        args.student_rows,
        args.output_dir,
        train_fraction=args.train_fraction,
        seed=args.seed,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
