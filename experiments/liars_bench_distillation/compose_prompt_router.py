#!/usr/bin/env python3
"""Compose cached general/specialist predictions with label-blind prompt routes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.evaluate_prompt_sweep import (
    split_records,
    summarize,
)
from experiments.liars_bench_distillation.router import (
    is_harm_pressure_choice_request,
    is_knowledge_report_request,
)


ROUTE_KINDS = ("knowledge", "choice", "union")


def parse_named_route(value: str) -> tuple[str, str]:
    """Parse NAME=knowledge|choice|union."""
    name, separator, kind = value.partition("=")
    if not separator or not name or kind not in ROUTE_KINDS:
        raise argparse.ArgumentTypeError(
            "expected NAME=knowledge|choice|union"
        )
    return name, kind


def route_matches(messages: Any, kind: str) -> bool:
    """Return whether an observable conversation contract activates a route."""
    knowledge = is_knowledge_report_request(messages)
    choice = is_harm_pressure_choice_request(messages)
    if kind == "knowledge":
        return knowledge
    if kind == "choice":
        return choice
    if kind == "union":
        return knowledge or choice
    raise ValueError(f"unsupported route kind: {kind!r}")


def load_predictions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load one aligned generation artifact."""
    return {
        (str(row["dataset"]), str(row["index"])): row
        for row in (
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    }


def compose_rows(
    records: list[dict[str, Any]],
    control: dict[tuple[str, str], dict[str, Any]],
    specialist: dict[tuple[str, str], dict[str, Any]],
    *,
    route_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace control predictions only where the frozen route activates."""
    rows = []
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for record in records:
        key = (str(record["dataset"]), str(record["index"]))
        if key not in control or key not in specialist:
            raise ValueError(f"missing prediction for {key}")
        routed = route_matches(record["messages"], route_kind)
        selected = specialist[key] if routed else control[key]
        row = dict(selected)
        row["routed"] = routed
        row["route_kind"] = route_kind if routed else None
        rows.append(row)
        if routed:
            category_counts[str(record["category"])] += 1
            source_counts[str(record["source_model"])] += 1
    return rows, {
        "rows": sum(category_counts.values()),
        "per_category": dict(sorted(category_counts.items())),
        "per_source_model": dict(sorted(source_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-artifact", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route", action="append", required=True, type=parse_named_route)
    parser.add_argument(
        "--split",
        choices=("development", "confirmation", "all"),
        required=True,
    )
    parser.add_argument("--split-seed", type=int, default=20260725)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    names = [name for name, _ in args.route]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate route names: {names}")

    all_records = [
        json.loads(line)
        for line in args.eval_artifact.read_text().splitlines()
        if line.strip()
    ]
    records = split_records(all_records, args.split, seed=args.split_seed)
    if args.expected_rows is not None and len(records) != args.expected_rows:
        raise ValueError(
            f"{args.split} contains {len(records)} rows, expected {args.expected_rows}"
        )
    control = load_predictions(args.control)
    specialist = load_predictions(args.specialist)
    control_rows, control_coverage = compose_rows(
        records,
        control,
        control,
        route_kind="union",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "split": args.split,
        "split_seed": args.split_seed,
        "rows": len(records),
        "conditions": {
            "control": summarize(control_rows, 0.0),
        },
        "routes": {},
    }
    for name, kind in args.route:
        rows, coverage = compose_rows(
            records,
            control,
            specialist,
            route_kind=kind,
        )
        with (args.output_dir / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result["conditions"][name] = summarize(rows, 0.0)
        result["routes"][name] = coverage
        print(
            name,
            json.dumps({
                "route": kind,
                "coverage": coverage,
                "metrics": result["conditions"][name],
            }),
            flush=True,
        )
    result["routes"]["control"] = {
        "rows": 0,
        "per_category": {},
        "per_source_model": {},
    }
    with (args.output_dir / "control.jsonl").open("w") as handle:
        for row in control_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
