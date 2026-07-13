#!/usr/bin/env python3
"""Format ranked Wikidata candidates for the frozen evidence judge sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def passage(row: dict[str, Any]) -> dict[str, str]:
    candidate = row.get("best_candidate")
    if not candidate:
        return {"title": "", "text": ""}
    return {"title": candidate["subject"], "text": candidate["fact"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load(args.input)
    passages = [passage(row) for row in rows]
    # A deterministic one-step rotation gives every row equally sized but
    # unrelated reference material. The sweep treats this as a noise control.
    shuffled = passages[1:] + passages[:1]
    output = []
    for row, real, noise in zip(rows, passages, shuffled, strict=True):
        output.append({
            "dataset": row["dataset"], "index": row["index"],
            "real_passages": [real] if real["text"] else [],
            "shuffled_passages": [noise] if noise["text"] else [],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(f"wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
