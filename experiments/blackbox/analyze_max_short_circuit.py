#!/usr/bin/env python3
"""Estimate exact max-ensemble positive short-circuit savings from a cache."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def analyze(path: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        grouped.setdefault((row["dataset"], row["index"]), []).append(row)
    if not grouped:
        raise ValueError("empty generation cache")

    evaluations = 0
    full_predictions = []
    short_predictions = []
    stop_counts: Counter[str] = Counter()
    member_order: list[str] | None = None
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["ensemble_member_index"]))
        names = [str(row["ensemble_member"]) for row in rows]
        if member_order is None:
            member_order = names
        if names != member_order:
            raise ValueError("ensemble member order differs across rows")
        ratings = [int(row.get("rating") or 1) for row in rows]
        full_prediction = int(max(ratings) >= 2)
        short_prediction = 0
        for offset, (name, rating) in enumerate(zip(names, ratings, strict=True), start=1):
            evaluations += 1
            if rating >= 2:
                short_prediction = 1
                stop_counts[name] += 1
                break
        else:
            stop_counts["all_negative"] += 1
        full_predictions.append(full_prediction)
        short_predictions.append(short_prediction)

    if full_predictions != short_predictions:
        raise AssertionError("short-circuit changed a cached max-ensemble decision")
    full_evaluations = len(grouped) * len(member_order or [])
    return {
        "n": len(grouped),
        "member_order": member_order,
        "full_prompt_evaluations": full_evaluations,
        "short_circuit_prompt_evaluations": evaluations,
        "mean_prompt_evaluations_per_row": evaluations / len(grouped),
        "prompt_evaluation_saving_fraction": 1 - evaluations / full_evaluations,
        "stop_counts": dict(stop_counts),
        "prediction_equivalent_on_cache": True,
        "caveat": "later-member batch composition changes and requires a generated equivalence run",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.generations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
