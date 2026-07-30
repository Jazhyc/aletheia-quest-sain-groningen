#!/usr/bin/env python3
"""Compose and validate the original V8 and Liars binary-soft caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-students", type=Path, required=True)
    parser.add_argument("--base-soft", type=Path, required=True)
    parser.add_argument("--liars-students", type=Path, required=True)
    parser.add_argument("--liars-soft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=13149)
    return parser.parse_args()


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"cache is empty: {path}")
    return rows


def keyed(rows: list[dict], name: str) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["index"]))
        if key in result:
            raise ValueError(f"duplicate {name} identity: {key}")
        result[key] = row
    return result


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    base_students = load(args.base_students)
    base_soft = load(args.base_soft)
    liars_students = load(args.liars_students)
    liars_soft = load(args.liars_soft)
    students = base_students + liars_students
    soft = base_soft + liars_soft
    if len(students) != args.expected_rows or len(soft) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} combined rows, got "
            f"students={len(students)} soft={len(soft)}"
        )
    student_by_key = keyed(students, "student")
    soft_by_key = keyed(soft, "soft")
    if set(student_by_key) != set(soft_by_key):
        raise ValueError("combined student and soft-target identities differ")
    for key, student in student_by_key.items():
        target = soft_by_key[key]
        if int(student["label"]) != int(target["label"]):
            raise ValueError(f"combined label mismatch for {key}")
        value = float(target["soft_target"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"combined soft target outside [0, 1] for {key}")
    ordered_keys = sorted(student_by_key)
    output_students = [student_by_key[key] for key in ordered_keys]
    output_soft = [soft_by_key[key] for key in ordered_keys]
    write(args.output_dir / "student_rows.jsonl", output_students)
    write(args.output_dir / "soft_targets.jsonl", output_soft)
    print(
        f"wrote combined cache rows={len(ordered_keys)} "
        f"base={len(base_students)} liars={len(liars_students)} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()

