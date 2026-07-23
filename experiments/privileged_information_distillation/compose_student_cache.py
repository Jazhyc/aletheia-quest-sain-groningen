#!/usr/bin/env python3
"""Compose student prompts from one teacher cache with targets from another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key(record: dict[str, Any]) -> tuple[str, Any]:
    return str(record.get("dataset", "")), record.get("index")


def compose_student_records(
    prompt_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    *,
    prompt_source: str,
    target_source: str,
) -> list[dict[str, Any]]:
    """Replace target-cache prompts by matching prompt-cache prompts."""
    prompts: dict[tuple[str, Any], dict[str, Any]] = {}
    for record in prompt_records:
        key = row_key(record)
        if key in prompts:
            raise ValueError(f"duplicate prompt-cache row: {key}")
        prompts[key] = record

    output = []
    seen: set[tuple[str, Any]] = set()
    for target in target_records:
        key = row_key(target)
        if key in seen:
            raise ValueError(f"duplicate target-cache row: {key}")
        seen.add(key)
        prompt = prompts.get(key)
        if prompt is None:
            raise ValueError(f"target row is missing from prompt cache: {key}")
        if int(prompt["label"]) != int(target["label"]):
            raise ValueError(f"label mismatch for {key}")
        if not prompt.get("student_prompt"):
            raise ValueError(f"empty student prompt for {key}")
        record = dict(target)
        record["student_prompt"] = prompt["student_prompt"]
        record["student_prompt_source"] = prompt_source
        record["student_target_source"] = target_source
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name-contains")
    args = parser.parse_args()

    prompt_cache = args.prompt_cache.resolve()
    target_cache = args.target_cache.resolve()
    prompt_records = read_jsonl(prompt_cache)
    target_records = read_jsonl(target_cache)
    if args.dataset_name_contains:
        target_records = [
            record for record in target_records
            if args.dataset_name_contains in str(record.get("dataset", ""))
        ]
    output = compose_student_records(
        prompt_records,
        target_records,
        prompt_source=prompt_cache.as_posix(),
        target_source=target_cache.as_posix(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record) + "\n" for record in output))
    usable = sum(
        not record.get("parse_error")
        and bool(record.get("label_match"))
        and bool(record.get("student_target"))
        for record in output
    )
    print(f"wrote {len(output)} rows ({usable} usable) to {args.output}")


if __name__ == "__main__":
    main()
