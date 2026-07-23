#!/usr/bin/env python3
"""Select false-supporting-detail positives with matched honest controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DETAIL_POSITIVE_RE = re.compile(
    r"(?:"
    r"\bcorrect(?:ly)?\b.{0,180}\b(?:but|however|although|yet)\b.{0,220}"
    r"\b(?:false|incorrect|inaccurate|misleading|fabricat)|"
    r"\b(?:but|however|although|yet)\b.{0,220}"
    r"\b(?:adds?|includes?|states?|claims?)\b.{0,100}"
    r"\b(?:false|incorrect|inaccurate|misleading|fabricat)|"
    r"\badds?\b.{0,80}\b(?:false|incorrect|inaccurate|misleading|fabricated)\b|"
    r"\b(?:false|incorrect|inaccurate|misleading)\b.{0,80}"
    r"\b(?:supporting|additional)\b.{0,40}\b(?:detail|claim)"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def stable_order(record: dict[str, Any], seed: int) -> str:
    """Return a deterministic pseudo-random sort key for a cached row."""
    key = f"{seed}\0{record.get('dataset', '')}\0{record.get('index', '')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def select_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    negative_ratio: float,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return hard positives and deterministic label-zero controls."""
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    usable = [
        record
        for record in records
        if not record.get("parse_error") and record.get("label_match")
    ]
    positives = [
        record
        for record in usable
        if int(record.get("label", -1)) == 1
        and DETAIL_POSITIVE_RE.search(str(record.get("reasoning_summary") or ""))
    ]
    negatives = sorted(
        (record for record in usable if int(record.get("label", -1)) == 0),
        key=lambda record: stable_order(record, seed),
    )
    negative_count = min(len(negatives), round(len(positives) * negative_ratio))
    selected = positives + negatives[:negative_count]
    selected.sort(key=lambda record: stable_order(record, seed + 1))
    return selected, len(positives), negative_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    selected, positives, negatives = select_records(
        records,
        seed=args.seed,
        negative_ratio=args.negative_ratio,
    )
    if not positives:
        raise RuntimeError("no false-supporting-detail positives matched")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(selected)} rows to {args.output}: "
        f"hard_positives={positives} honest_controls={negatives}"
    )


if __name__ == "__main__":
    main()
