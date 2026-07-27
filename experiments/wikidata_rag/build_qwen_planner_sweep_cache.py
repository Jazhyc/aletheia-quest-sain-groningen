#!/usr/bin/env python3
"""Build real/shuffled evidence conditions from frozen Qwen planner selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.wikidata_rag.build_cross_encoder_sweep_cache import (
    cross_dataset_donors,
    passage,
)


def load(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    """Render all validated selections as one bounded evidence payload."""
    if row.get("parse_error") or not row.get("selected"):
        return None
    candidates = {str(item["id"]): item for item in row.get("candidates", [])}
    selections = []
    for plan in row["selected"]:
        candidate = candidates.get(str(plan["id"]))
        if candidate is None:
            continue
        selections.append(dict(candidate) | {
            "planner_relation": str(plan["relation"]),
            "claim_quote": str(plan["claim_quote"]),
        })
    if not selections:
        return None
    if len(selections) == 1:
        return selections[0]
    return {
        "subject": "Selected Wikidata facts",
        "fact": "\n".join(
            f"- {item['subject']}: {item['fact']}" for item in selections
        ),
        "selections": selections,
    }


def build_cache(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach matched cross-dataset controls only to planner-active rows."""
    selected = [selected_candidate(row) for row in rows]
    shuffled = cross_dataset_donors(rows, selected)
    records = []
    for row, real, noise in zip(rows, selected, shuffled, strict=True):
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
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = build_cache(load(args.input))
    active = sum(bool(row["real_passages"]) for row in records)
    controls = sum(bool(row["shuffled_passages"]) for row in records)
    if active and controls != active:
        raise RuntimeError(
            f"cross-dataset controls missing for {active - controls}/{active} active rows"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"
    )
    print(
        f"wrote {len(records)} rows with {active} planner facts and "
        f"{controls} cross-dataset controls",
        flush=True,
    )


if __name__ == "__main__":
    main()
