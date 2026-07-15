#!/usr/bin/env python3
"""Audit whether teacher-only retrieval changes supervision without prompt leakage."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, Any], dict[str, Any]]:
    output = {(str(row["dataset"]), row["index"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("duplicate dataset/index rows")
    return output


def content_words(text: str) -> set[str]:
    return {word.lower() for word in WORD_RE.findall(text) if len(word) >= 4}


def passage_text(passages: list[dict[str, Any]]) -> str:
    return " ".join(
        f"{passage.get('title', '')} {passage.get('text', '')}" for passage in passages
    )


def audit_teacher_caches(
    baseline_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    shuffled_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    baseline = keyed(baseline_rows)
    real = keyed(real_rows)
    shuffled = keyed(shuffled_rows)
    references = keyed(reference_rows)
    keys = sorted(real)
    if set(shuffled) != set(real):
        raise ValueError("real and shuffled teacher caches cover different rows")
    if not set(keys).issubset(baseline):
        raise ValueError("baseline teacher cache is missing comparison rows")
    if not set(keys).issubset(references):
        raise ValueError("retrieval cache is missing comparison rows")

    counts: Counter[str] = Counter()
    samples = []
    for key in keys:
        base_row = baseline[key]
        real_row = real[key]
        shuffled_row = shuffled[key]
        reference = references[key]
        if not (
            base_row["label"] == real_row["label"] == shuffled_row["label"]
            and base_row["student_prompt"]
            == real_row["student_prompt"]
            == shuffled_row["student_prompt"]
        ):
            raise ValueError(f"non-retrieval inputs differ for {key}")
        counts["rows"] += 1
        counts["real_usable"] += int(not real_row.get("parse_error") and real_row.get("label_match"))
        counts["shuffled_usable"] += int(
            not shuffled_row.get("parse_error") and shuffled_row.get("label_match")
        )
        counts["student_reference_leaks"] += int(
            "<reference_material>" in str(real_row["student_prompt"])
            or "<reference_material>" in str(shuffled_row["student_prompt"])
        )
        real_passages = list(reference.get("real_passages") or [])
        shuffled_passages = list(reference.get("shuffled_passages") or [])
        if not real_passages:
            continue
        counts["active_rows"] += 1
        base_summary = str(base_row.get("reasoning_summary") or "")
        real_summary = str(real_row.get("reasoning_summary") or "")
        shuffled_summary = str(shuffled_row.get("reasoning_summary") or "")
        counts["real_changed_from_baseline"] += int(real_summary != base_summary)
        counts["shuffled_changed_from_baseline"] += int(shuffled_summary != base_summary)
        counts["real_changed_from_shuffled"] += int(real_summary != shuffled_summary)
        real_new = content_words(real_summary) - content_words(base_summary)
        shuffled_new = content_words(shuffled_summary) - content_words(base_summary)
        real_source = content_words(passage_text(real_passages))
        shuffled_source = content_words(passage_text(shuffled_passages))
        counts["real_introduces_source_term"] += int(bool(real_new & real_source))
        counts["shuffled_introduces_source_term"] += int(
            bool(shuffled_new & shuffled_source)
        )
        counts["real_mentions_retrieval"] += int(
            bool({"retrieval", "retrieved", "reference"} & content_words(real_summary))
        )
        if real_summary != base_summary and len(samples) < sample_limit:
            samples.append({
                "dataset": key[0],
                "index": key[1],
                "label": real_row["label"],
                "real_passages": real_passages,
                "baseline_summary": base_summary,
                "real_summary": real_summary,
                "shuffled_summary": shuffled_summary,
            })
    return {"counts": dict(counts), "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()
    report = audit_teacher_caches(
        read_jsonl(args.baseline),
        read_jsonl(args.real),
        read_jsonl(args.shuffled),
        read_jsonl(args.reference),
        sample_limit=args.sample_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
