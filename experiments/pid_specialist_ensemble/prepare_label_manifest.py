#!/usr/bin/env python3
"""Select a deterministic training manifest directly from split labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.train_student_sft import (
    select_stratified_fraction,
)
from experiments.qwen_grpo_lora.evaluate_qwen_grpo_lora import (
    load_labels,
    load_split_config,
)


def load_label_records(
    split_config: Path, dataset_name_contains: str
) -> list[dict]:
    records = []
    for config in load_split_config(split_config, ROOT):
        if dataset_name_contains not in config.name:
            continue
        labels = load_labels(config)
        records.extend(
            {
                "dataset": config.name,
                "index": index,
                "label": int(label),
            }
            for index, label in zip(
                labels["index"], labels["label"], strict=True
            )
        )
    if not records:
        raise RuntimeError("no matching label rows")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--dataset-name-contains", default="varied-deception")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_label_records(
        args.split_config.resolve(), args.dataset_name_contains
    )
    selected = select_stratified_fraction(records, args.fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in selected
    ))
    report = {
        "source": args.split_config.resolve().as_posix(),
        "dataset_name_contains": args.dataset_name_contains,
        "raw_rows": len(records),
        "selected_rows": len(selected),
        "fraction": args.fraction,
        "seed": args.seed,
        "selected_by_label": dict(sorted(Counter(
            record["label"] for record in selected
        ).items())),
        "selected_by_stratum": {
            json.dumps([dataset, label]): count
            for (dataset, label), count in sorted(Counter(
                (record["dataset"], record["label"])
                for record in selected
            ).items())
        },
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
