#!/usr/bin/env python3
"""Freeze the usable varied-only row keys from a PI teacher cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def usable_manifest(
    records: list[dict[str, Any]],
    dataset_name_contains: str,
) -> list[dict[str, Any]]:
    """Return dataset/index/label rows accepted by the original student."""
    return [
        {
            "dataset": record["dataset"],
            "index": record["index"],
            "label": int(record["label"]),
        }
        for record in records
        if dataset_name_contains in str(record.get("dataset", ""))
        and not record.get("parse_error")
        and record.get("label_match") is True
        and record.get("student_target")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--dataset-name-contains",
        default="varied-deception",
    )
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    manifest = usable_manifest(records, args.dataset_name_contains)
    if args.expected_rows is not None and len(manifest) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} usable rows, found {len(manifest)}"
        )
    if len({
        (row["dataset"], str(row["index"]))
        for row in manifest
    }) != len(manifest):
        raise ValueError("usable manifest contains duplicate dataset/index keys")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in manifest
        )
    )
    print(f"wrote {len(manifest)} rows to {args.output}")


if __name__ == "__main__":
    main()
