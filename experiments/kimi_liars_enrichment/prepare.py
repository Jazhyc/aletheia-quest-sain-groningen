#!/usr/bin/env python3
"""Select a frozen, balanced Liars' Bench enrichment pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.prepare_teacher_data import (
    exclusion_ids,
    load_category,
    stable_sample,
    write_jsonl,
)
from experiments.privileged_information_distillation.core import build_student_prompt
from experiments.kimi_liars_enrichment.prompts import CATEGORY_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liars-root", type=Path, required=True)
    parser.add_argument("--pilot-artifact", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def selected_conditions(path: Path) -> dict[str, str]:
    audit = json.loads(path.read_text())
    selected = {
        category: str(result["selected_condition"])
        for category, result in audit["categories"].items()
        if result["admit"]
    }
    if not selected:
        raise ValueError("pilot admitted zero Liars' Bench categories")
    return selected


def build_record(row: dict, spec: dict, teacher_condition: str) -> dict:
    return {
        "dataset": f"liars-bench/{row['category']}",
        "index": str(row["sample_id"]),
        "category": str(row["category"]),
        "source_model": str(row["source_model"]),
        "label": int(row["label"]),
        "messages": row["messages"],
        "student_prompt": build_student_prompt(
            row["messages"],
            str(spec["student_prompt"]),
            int(spec["student_max_chars"]),
            str(spec["student_truncation"]),
        ),
        # Prediction-only soft training normally replaces cached prompt prose
        # with the global config prompt. This explicit per-source template keeps
        # Phoenix 8's deployed HP-KR/action routes training-matched.
        "_source_prompt_template": str(spec["student_prompt"]),
        "student_target": "<unused_direct_soft_target>",
        "parse_error": False,
        "label_match": True,
        "source": "liars_bench_plus_kimi_k3_semantic_binary_soft_target",
        "teacher_prompt_kind": teacher_condition,
    }


def main() -> None:
    args = parse_args()
    conditions = selected_conditions(args.audit)
    excluded = exclusion_ids([args.pilot_artifact])
    parts: list[pd.DataFrame] = []
    for offset, category in enumerate(CATEGORY_SPECS):
        if category not in conditions:
            continue
        spec = CATEGORY_SPECS[category]
        frame = load_category(category, args.liars_root)
        selected_frame = stable_sample(
            frame,
            per_label=int(spec["train_per_label"]),
            seed=args.seed + offset,
            excluded_ids=excluded,
        )
        overlap = set(selected_frame["sample_id"]) & excluded
        if overlap:
            raise RuntimeError(f"pilot overlap in {category}: {sorted(overlap)[:3]}")
        parts.append(selected_frame)
        print(
            f"category={category} selected={len(selected_frame)} "
            f"labels={selected_frame['label'].value_counts().sort_index().to_dict()} "
            f"models={selected_frame['source_model'].value_counts().to_dict()}",
            flush=True,
        )
    frame = pd.concat(parts, ignore_index=True)
    records = [
        build_record(
            row,
            CATEGORY_SPECS[str(row["category"])],
            conditions[str(row["category"])],
        )
        for row in frame.to_dict(orient="records")
    ]
    identities = {(row["dataset"], row["index"]) for row in records}
    if len(identities) != len(records):
        raise ValueError("selected dataset/index identities are not unique")
    write_jsonl(args.output, records)
    print(
        f"wrote={args.output} rows={len(records)} "
        f"selected_conditions={conditions}",
        flush=True,
    )


if __name__ == "__main__":
    main()
