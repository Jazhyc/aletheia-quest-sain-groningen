#!/usr/bin/env python3
"""Audit frozen semantic prompt routes over every local competition split."""

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

from experiments.blackbox.run_judge import (
    load_examples_for_labels,
    load_label_subset,
    load_split_config,
)
from experiments.liars_bench_distillation.compose_prompt_router import route_matches
from experiments.liars_bench_distillation.router import (
    is_action_report_conversation,
)


ROUTES = ("knowledge", "choice", "union", "action")


def matches(messages: Any, route: str) -> bool:
    """Dispatch legacy prompt routes plus the Phoenix 4.3 action route."""
    if route == "action":
        return is_action_report_conversation(messages)
    return route_matches(messages, route)


def coverage(examples: list[dict[str, Any]], route: str) -> dict[str, Any]:
    """Summarize label-blind route coverage without inspecting outcomes."""
    keys = []
    datasets: Counter[str] = Counter()
    for example in examples:
        if matches(example["messages"], route):
            keys.append([str(example["dataset"]), example["index"]])
            datasets[str(example["dataset"])] += 1
    return {
        "rows": len(keys),
        "per_dataset": dict(sorted(datasets.items())),
        "routed_keys": keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {"splits": {}, "rows": 0}
    for split in ("train", "validation", "test"):
        configs = load_split_config(args.splits_dir / f"dry.{split}.yaml", ROOT)
        examples = []
        for config in configs:
            labels = load_label_subset(config)
            loaded = load_examples_for_labels(config.name, labels, None)
            for example in loaded:
                examples.append({**example, "dataset": config.name})
        result["splits"][split] = {
            "rows": len(examples),
            "routes": {
                route: coverage(examples, route)
                for route in ROUTES
            },
        }
        result["rows"] += len(examples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
