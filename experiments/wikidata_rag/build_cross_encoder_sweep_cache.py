#!/usr/bin/env python3
"""Build real/shuffled evidence conditions from a frozen MiniLM refit artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_validation_candidates(
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    predictions: np.ndarray,
    slices: np.ndarray,
    threshold: float,
) -> list[dict[str, Any] | None]:
    all_rows = training_rows + validation_rows
    if len(slices) != len(all_rows):
        raise ValueError(f"prediction rows {len(slices)} != utility rows {len(all_rows)}")
    selected = []
    for row_number, row in enumerate(validation_rows, start=len(training_rows)):
        start, end = (int(value) for value in slices[row_number])
        if end - start != len(row["candidates"]):
            raise ValueError(
                f"candidate count mismatch at {row['dataset']}:{row['index']}: "
                f"predictions={end - start} cache={len(row['candidates'])}"
            )
        if start == end:
            selected.append(None)
            continue
        local = int(np.argmax(predictions[start:end]))
        score = float(predictions[start + local])
        candidate = row["candidates"][local]
        selected.append(
            dict(candidate) | {"cross_encoder_score": score}
            if score >= threshold else None
        )
    return selected


def cross_dataset_donors(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any] | None],
) -> list[dict[str, Any] | None]:
    """Choose deterministic selected donors from other dataset units."""
    active = [index for index, candidate in enumerate(selected) if candidate is not None]
    donors: list[dict[str, Any] | None] = [None] * len(rows)
    for position, recipient_index in enumerate(active):
        recipient = rows[recipient_index]
        for offset in range(1, len(active) + 1):
            donor_index = active[(position + offset) % len(active)]
            donor = rows[donor_index]
            if donor["dataset"] != recipient["dataset"]:
                donors[recipient_index] = dict(selected[donor_index] or {}) | {
                    "donor_dataset": donor["dataset"],
                    "donor_index": donor["index"],
                }
                break
    return donors


def passage(candidate: dict[str, Any] | None) -> list[dict[str, str]]:
    if not candidate:
        return []
    return [{
        "title": str(candidate.get("subject", "Unknown subject")),
        "text": str(candidate.get("fact", "")),
    }]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training_rows = load_jsonl(args.training_input)
    validation_rows = load_jsonl(args.validation_input)
    artifact = np.load(args.predictions)
    policy_predictions = artifact["policy_predictions"].astype(np.float32)
    slices = artifact["row_slices"].astype(np.int64)
    threshold = float(artifact["threshold"].reshape(-1)[0])
    selected = select_validation_candidates(
        training_rows, validation_rows, policy_predictions, slices, threshold
    )
    shuffled = cross_dataset_donors(validation_rows, selected)
    records = []
    for row, real, noise in zip(validation_rows, selected, shuffled, strict=True):
        records.append({
            "dataset": row["dataset"],
            "index": row["index"],
            "real_passages": passage(real),
            "shuffled_passages": passage(noise),
            "selected_candidate": real,
            "shuffled_from": (
                {
                    "dataset": noise["donor_dataset"],
                    "index": noise["donor_index"],
                }
                if noise else None
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    active = sum(candidate is not None for candidate in selected)
    matched_noise = sum(candidate is not None for candidate in shuffled)
    print(
        f"wrote {len(records)} rows with {active} selected facts and "
        f"{matched_noise} cross-dataset controls at threshold {threshold:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
