#!/usr/bin/env python3
"""Audit the passage/Boolean router over all local competition splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.run_judge import (
    load_examples_for_labels,
    load_label_subset,
    load_split_config,
)
from experiments.liars_bench_distillation.router import is_passage_true_false_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {"splits": {}, "rows": 0, "routed": 0}
    for split in ("train", "validation", "test"):
        configs = load_split_config(args.splits_dir / f"dry.{split}.yaml", ROOT)
        rows = routed = 0
        routed_keys = []
        for config in configs:
            labels = load_label_subset(config)
            examples = load_examples_for_labels(config.name, labels, None)
            rows += len(examples)
            for example in examples:
                if is_passage_true_false_request(example["messages"]):
                    routed += 1
                    routed_keys.append([config.name, example["index"]])
        result["splits"][split] = {
            "rows": rows,
            "routed": routed,
            "routed_keys": routed_keys,
        }
        result["rows"] += rows
        result["routed"] += routed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
