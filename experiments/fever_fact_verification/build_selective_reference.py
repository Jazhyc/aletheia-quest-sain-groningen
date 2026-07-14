#!/usr/bin/env python3
"""Build a reader cache containing only GPT-OSS-audited decisive evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.build_teacher_reference import (
    limit_reference_passages,
)
from experiments.fever_fact_verification.selective_hop import DECISIVE, claim_key


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_passage(row: dict[str, Any]) -> dict[str, Any] | None:
    candidate = row.get("selected_candidate")
    if row.get("decision") not in DECISIVE or not candidate:
        return None
    return {
        "title": str(candidate.get("title") or "Wikipedia"),
        "text": (
            f"Claim: {row['proposition']}\n"
            f"Source sentence: {candidate.get('text', '')}"
        ),
        "url": str(candidate.get("url") or ""),
        "claim_index": int(row["claim_index"]),
        "audit_relation": str(row["decision"]),
        "audit_stage": str(row.get("stage") or ""),
    }


def build_cache(
    initial: list[dict[str, Any]],
    second: list[dict[str, Any]],
    universe: list[dict[str, Any]],
    *,
    max_reference_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    second_by_key: dict[tuple[str, Any, int], dict[str, Any]] = {}
    for row in second:
        key = claim_key(row)
        current = second_by_key.get(key)
        if current is None or (
            current.get("decision") not in DECISIVE
            and row.get("decision") in DECISIVE
        ):
            # Follow-up files are ordered from cheapest to most expensive. Keep
            # the first decisive result and consult later stages only after an
            # abstention.
            second_by_key[key] = row
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    stages = Counter()
    decisions = Counter()
    for first in initial:
        chosen = first
        if first.get("decision") not in DECISIVE:
            chosen = second_by_key.get(claim_key(first), first)
        decisions[str(chosen.get("decision") or "ABSTAIN")] += 1
        passage = selected_passage(chosen)
        if passage is None:
            continue
        grouped[(str(first["dataset"]), first["index"])].append(passage)
        stages[str(chosen.get("stage") or "initial")] += 1
    grouped = {
        key: limit_reference_passages(passages, max_reference_chars)
        for key, passages in grouped.items()
    }
    grouped = {key: passages for key, passages in grouped.items() if passages}
    active = sorted(grouped)
    donors = {}
    for position, key in enumerate(active):
        for offset in range(len(active)):
            candidate = active[(position + len(active) // 2 + offset) % len(active)]
            if candidate != key and candidate[0] != key[0]:
                donors[key] = candidate
                break
        if key not in donors:
            raise ValueError(f"no cross-dataset donor for {key}")
    keys = sorted({(str(row["dataset"]), row["index"]) for row in universe})
    output = []
    for dataset, index in keys:
        key = dataset, index
        passages = grouped.get(key, [])
        donor = donors.get(key)
        output.append({
            "dataset": dataset,
            "index": index,
            "passages": passages,
            "real_passages": passages,
            "shuffled_passages": grouped[donor] if donor is not None else [],
            "shuffled_donor_dataset": donor[0] if donor is not None else None,
            "shuffled_donor_index": donor[1] if donor is not None else None,
        })
    return output, {
        "rows": len(output),
        "active_rows": len(active),
        "passages": sum(len(row["passages"]) for row in output),
        "decisions": decisions,
        "selected_stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-audit", type=Path, required=True)
    parser.add_argument("--second-audit", type=Path, action="append", required=True)
    parser.add_argument("--row-universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-reference-chars", type=int, default=5000)
    args = parser.parse_args()
    output, summary = build_cache(
        read_jsonl(args.initial_audit),
        [
            {**row, "stage": path.stem}
            for path in args.second_audit
            for row in read_jsonl(path)
        ],
        read_jsonl(args.row_universe),
        max_reference_chars=args.max_reference_chars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
