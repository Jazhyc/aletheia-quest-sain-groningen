#!/usr/bin/env python3
"""Build label-audited student prompt rows without a rationale teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import build_student_prompt
from experiments.qwen_grpo_lora.run_qwen_grpo_lora import (
    load_labels,
    load_split_config,
)


def load_soft_targets(path: Path) -> dict[tuple[str, Any], tuple[int, float]]:
    """Load unique binary soft targets keyed by dataset and row index."""
    targets: dict[tuple[str, Any], tuple[int, float]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["dataset"]), record["index"])
        if key in targets:
            raise ValueError(f"duplicate soft target at line {line_number}: {key}")
        target = float(record["soft_target"])
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"soft target outside [0, 1] for {key}: {target}")
        targets[key] = (int(record["label"]), target)
    if not targets:
        raise ValueError(f"soft-target artifact is empty: {path}")
    return targets


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config-name",
        default="pid_qwen27_tvg_binary_soft_distillation_v1",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-name-contains", default="varied-deception")
    parser.add_argument("--expected-rows", type=int, default=2880)
    parser.add_argument(
        "--source",
        default="public_split_plus_qwen27_binary_soft_target",
        help="Provenance label written to every student row.",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    with initialize_config_dir(
        version_base=None,
        config_dir=(root / "configs").as_posix(),
    ):
        cfg = compose(config_name=args.config_name)

    targets = load_soft_targets(args.soft_targets)
    split_path = root / str(cfg.splits_dir) / f"dry.{args.split}.yaml"
    dataset_configs = load_split_config(split_path, root)
    rows: list[dict[str, Any]] = []
    found: set[tuple[str, Any]] = set()
    for dataset_cfg in dataset_configs:
        if args.dataset_name_contains not in dataset_cfg.name:
            continue
        labels = load_labels(dataset_cfg)
        label_by_index = dict(zip(labels["index"], labels["label"], strict=True))
        dataset = load_dataset(dataset_cfg.name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        for example in dataset:
            index = example["index"]
            if index not in label_by_index:
                continue
            key = (dataset_cfg.name, index)
            if key not in targets:
                raise ValueError(f"soft-target artifact is missing public row {key}")
            label = int(label_by_index[index])
            target_label, _ = targets[key]
            if target_label != label:
                raise ValueError(
                    f"soft-target label mismatch for {key}: {target_label} != {label}"
                )
            rows.append({
                "dataset": dataset_cfg.name,
                "index": index,
                "label": label,
                "student_prompt": build_student_prompt(
                    example["messages"],
                    str(cfg.student.prompt),
                    int(cfg.student.max_prompt_chars),
                    str(cfg.student.context_truncation),
                    include_reasoning=False,
                ),
                # The trainer's direct-only configuration never evaluates this
                # sentinel completion; it keeps the general cache schema valid.
                "student_target": "<unused_direct_soft_target>",
                "parse_error": False,
                "label_match": True,
                "source": args.source,
            })
            found.add(key)

    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} student rows, got {len(rows)}")
    extra_targets = sorted(set(targets) - found)
    if extra_targets:
        raise ValueError(
            f"soft-target artifact has {len(extra_targets)} unmatched rows; "
            f"first={extra_targets[0]}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in rows)
    )
    print(f"wrote {len(rows)} public student rows to {args.output}")


if __name__ == "__main__":
    main()
