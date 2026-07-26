#!/usr/bin/env python3
"""Build a labels-only split for incomplete Qwen-27B D/K/S trace rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


EXPECTED_MEMBERS = {"details4096", "known4096", "scrutiny4096"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def failed_row_keys(
    base_rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> set[tuple[str, Any]]:
    """Return usable base rows lacking three parsed, closed trace members."""
    usable = {
        (str(row["dataset"]), row["index"])
        for row in base_rows
        if not row.get("parse_error")
        and row.get("label_match")
        and "varied-deception" in str(row.get("dataset", ""))
    }
    members: dict[tuple[str, Any], set[str]] = defaultdict(set)
    for record in generations:
        key = (str(record["dataset"]), record["index"])
        if (
            key in usable
            and not record.get("parse_error")
            and "</think>" in str(record.get("text", ""))
        ):
            members[key].add(str(record["ensemble_member"]))
    return {
        key for key in usable if members.get(key, set()) != EXPECTED_MEMBERS
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_rows = load_jsonl(args.base_teacher)
    failures = failed_row_keys(base_rows, load_jsonl(args.generations))
    if not args.minimum_rows <= len(failures) <= args.maximum_rows:
        raise ValueError(
            f"retry row count {len(failures)} outside "
            f"[{args.minimum_rows}, {args.maximum_rows}]"
        )
    labels = {
        (str(row["dataset"]), row["index"]): int(row["label"])
        for row in base_rows
        if (str(row["dataset"]), row["index"]) in failures
    }
    by_dataset: dict[str, list[tuple[Any, int]]] = defaultdict(list)
    for (dataset, index), label in labels.items():
        by_dataset[dataset].append((index, label))

    labels_dir = args.output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for dataset, rows in sorted(by_dataset.items()):
        path = labels_dir / (dataset.replace("/", "__") + ".csv")
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "deceptive"])
            writer.writeheader()
            for index, label in sorted(rows, key=lambda item: str(item[0])):
                writer.writerow({"index": index, "deceptive": bool(label)})
        datasets.append({
            "name": dataset,
            "labels_uri": path.resolve().as_posix(),
            "id_column": "index",
            "label_column": "deceptive",
        })
    (args.output_dir / "dry.train.yaml").write_text(
        yaml.safe_dump({"datasets": datasets}, sort_keys=False)
    )
    audit = {
        "retry_rows": len(failures),
        "datasets": {name: len(rows) for name, rows in sorted(by_dataset.items())},
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--base-teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=1)
    parser.add_argument("--maximum-rows", type=int, default=600)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
