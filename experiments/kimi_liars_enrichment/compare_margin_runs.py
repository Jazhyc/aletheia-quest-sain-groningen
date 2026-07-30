#!/usr/bin/env python3
"""Compare aligned continuous-margin runs and enforce an exact parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def load_scores(path: Path, column: str) -> dict[tuple[str, str], float]:
    """Load one score per dataset/index identity."""
    scores: dict[tuple[str, str], float] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        identity = (str(row["dataset"]), str(row["index"]))
        if identity in scores:
            raise ValueError(f"duplicate identity at {path}:{line_number}: {identity}")
        scores[identity] = float(row[column])
    if not scores:
        raise ValueError(f"no scores in {path}")
    return scores


def float32_digest(values: np.ndarray) -> str:
    """Hash sorted-identity float32 scores in little-endian representation."""
    digest = hashlib.sha256()
    for value in values.astype(np.float32):
        digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


def compare(
    reference: dict[tuple[str, str], float],
    candidate: dict[tuple[str, str], float],
) -> dict[str, object]:
    """Return exact-score and ordering parity diagnostics."""
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(
            f"identity mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    identities = sorted(reference)
    left = np.asarray([reference[key] for key in identities], dtype=np.float64)
    right = np.asarray([candidate[key] for key in identities], dtype=np.float64)
    differences = np.abs(left - right)
    left_order = np.sign(left[:, None] - left[None, :])
    right_order = np.sign(right[:, None] - right[None, :])
    upper = np.triu_indices(len(left), k=1)
    ordering_changes = int(
        np.count_nonzero(left_order[upper] != right_order[upper])
    )
    return {
        "rows": len(left),
        "exact_scores": int(np.count_nonzero(left == right)),
        "mean_absolute_difference": float(differences.mean()),
        "maximum_absolute_difference": float(differences.max()),
        "ordering_changes": ordering_changes,
        "reference_unique_scores": int(np.unique(left).size),
        "candidate_unique_scores": int(np.unique(right).size),
        "reference_float32_digest": float32_digest(left),
        "candidate_float32_digest": float32_digest(right),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--score-column", default="direct_margin_score")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    result = compare(
        load_scores(args.reference, args.score_column),
        load_scores(args.candidate, args.score_column),
    )
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if args.require_exact and (
        result["exact_scores"] != result["rows"]
        or result["ordering_changes"] != 0
        or result["reference_float32_digest"]
        != result["candidate_float32_digest"]
    ):
        raise SystemExit("exact BF16 parity gate failed")


if __name__ == "__main__":
    main()
