#!/usr/bin/env python3
"""Expose one retrieval-query mode through the standard real/shuffled fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "atomic"), required=True)
    args = parser.parse_args()

    rows = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({
            **row,
            "real_passages": row[f"{args.mode}_real_passages"],
            "shuffled_passages": row[f"{args.mode}_shuffled_passages"],
            "selected_query_mode": args.mode,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    print(json.dumps({
        "mode": args.mode,
        "rows": len(rows),
        "active_rows": sum(bool(row["real_passages"]) for row in rows),
        "passages": sum(len(row["real_passages"]) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
