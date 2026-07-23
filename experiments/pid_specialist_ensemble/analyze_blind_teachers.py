#!/usr/bin/env python3
"""Evaluate genuine blind-teacher decisions without filtering their mistakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_ensemble import (
    load_member_frame,
    metrics,
    parse_member,
)


KEYS = ["dataset", "index"]


def load_teacher_frame(specs: list[tuple[str, Path]]) -> pd.DataFrame:
    merged = None
    for name, path in specs:
        frame = pd.read_json(path, lines=True)
        frame["index"] = frame["index"].astype(str)
        frame[name] = [
            0.0 if parse_error or prediction not in (0, 1) else float(prediction)
            for parse_error, prediction in zip(
                frame["parse_error"], frame["prediction"], strict=True
            )
        ]
        frame = frame[[*KEYS, "label", name, "parse_error"]].rename(columns={
            "label": f"label_{name}",
            "parse_error": f"parse_error_{name}",
        })
        merged = frame if merged is None else merged.merge(
            frame, on=KEYS, how="outer", validate="one_to_one"
        )
    if merged is None:
        raise ValueError("at least one teacher member is required")
    label_columns = [column for column in merged if column.startswith("label_")]
    if merged[label_columns].isna().any().any():
        raise ValueError("teacher caches do not cover identical row keys")
    if not merged[label_columns].nunique(axis=1).eq(1).all():
        raise ValueError("teacher caches disagree on labels")
    merged["label"] = merged.pop(label_columns[0]).astype(int)
    return merged.drop(columns=label_columns[1:]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--student-member", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    specs = [parse_member(value) for value in args.member]
    names = [name for name, _ in specs]
    frame = load_teacher_frame(specs)
    features = frame[names].to_numpy(dtype=float)
    scores = {
        **{name: frame[name].to_numpy(dtype=float) for name in names},
        "or": features.max(axis=1),
        "majority": (
            features.sum(axis=1) >= len(names) // 2 + 1
        ).astype(float),
        "unanimity": (features.sum(axis=1) == len(names)).astype(float),
    }
    report = {
        "members": names,
        "rows": len(frame),
        "parse_errors": {
            name: int(frame[f"parse_error_{name}"].sum()) for name in names
        },
        "pairwise_disagreement": {
            f"{left}__{right}": int((frame[left] != frame[right]).sum())
            for position, left in enumerate(names)
            for right in names[position + 1:]
        },
        "metrics": {
            name: metrics(frame, values) for name, values in scores.items()
        },
    }
    if args.student_member:
        student_specs = [parse_member(value) for value in args.student_member]
        if [name for name, _ in student_specs] != names:
            raise ValueError("teacher and student member names/order must match")
        students = load_member_frame(student_specs)
        student_columns = [*KEYS, *names]
        students = students[student_columns].rename(columns={
            name: f"student_{name}" for name in names
        })
        aligned = frame.merge(
            students, on=KEYS, validate="one_to_one"
        )
        if len(aligned) != len(frame):
            raise ValueError("teacher and student artifacts cover different rows")
        report["teacher_student_agreement"] = {
            name: {
                "agreements": int(
                    (aligned[name] == aligned[f"student_{name}"]).sum()
                ),
                "disagreements": int(
                    (aligned[name] != aligned[f"student_{name}"]).sum()
                ),
                "agreement_rate": float(
                    (aligned[name] == aligned[f"student_{name}"]).mean()
                ),
            }
            for name in names
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
