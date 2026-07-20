#!/usr/bin/env python3
"""Build a compact SQLite FTS index from frozen Wikipedia page extracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compact_wikipedia_index.core import create_index, iter_page_sentences


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-chars", type=int, default=32)
    parser.add_argument("--maximum-chars", type=int, default=900)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    pages = read_jsonl(args.pages)
    with sqlite3.connect(args.output) as connection:
        report = create_index(
            connection,
            iter_page_sentences(
                pages,
                minimum_chars=args.minimum_chars,
                maximum_chars=args.maximum_chars,
            ),
        )
        connection.execute("VACUUM")
    report.update({
        "source_rows": len(pages),
        "database_bytes": args.output.stat().st_size,
        "source": args.pages.as_posix(),
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
