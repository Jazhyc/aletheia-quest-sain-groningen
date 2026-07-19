#!/usr/bin/env python3
"""Compare matched specialist decisions across two LoRA ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pid_specialist_ensemble.analyze_ensemble import (
    KEYS,
    comparison_counts,
    load_member_frame,
    parse_member,
)


def compare_rank_frames(rank1, rank4, names: list[str]) -> dict:
    """Align two matched member frames and count rank-4 fixes and breaks."""
    rank1 = rank1[[*KEYS, "label", *names]].rename(columns={
        name: f"rank1_{name}" for name in names
    })
    rank4 = rank4[[*KEYS, "label", *names]].rename(columns={
        "label": "rank4_label",
        **{name: f"rank4_{name}" for name in names},
    })
    aligned = rank1.merge(rank4, on=KEYS, validate="one_to_one")
    if len(aligned) != len(rank1) or len(aligned) != len(rank4):
        raise ValueError("rank artifacts cover different rows")
    if not (aligned["label"] == aligned["rank4_label"]).all():
        raise ValueError("rank artifacts disagree on labels")

    labels = aligned["label"].to_numpy(dtype=int)
    return {
        "members": names,
        "rows": len(aligned),
        "rank4_vs_rank1": {
            name: comparison_counts(
                labels,
                aligned[f"rank1_{name}"].to_numpy(dtype=float),
                aligned[f"rank4_{name}"].to_numpy(dtype=float),
            )
            for name in names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank1-member", action="append", required=True)
    parser.add_argument("--rank4-member", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rank1_specs = [parse_member(value) for value in args.rank1_member]
    rank4_specs = [parse_member(value) for value in args.rank4_member]
    rank1_names = [name for name, _ in rank1_specs]
    rank4_names = [name for name, _ in rank4_specs]
    if rank1_names != rank4_names:
        raise ValueError("rank-1 and rank-4 member names/order must match")

    rank1 = load_member_frame(rank1_specs)
    rank4 = load_member_frame(rank4_specs)
    names = rank1_names
    report = compare_rank_frames(rank1, rank4, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
