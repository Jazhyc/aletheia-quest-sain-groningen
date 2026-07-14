#!/usr/bin/env python3
"""Convert claim-level FEVER results into row-level privileged references."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def truncate_chars(value: str, limit: int) -> str:
    """Bound pathological list-like Wikipedia sentences without losing provenance."""
    if len(value) <= limit:
        return value
    marker = " ..."
    if limit <= len(marker):
        return value[:limit]
    return value[: limit - len(marker)].rstrip() + marker


def limit_reference_passages(
    passages: list[dict[str, Any]], max_chars: int
) -> list[dict[str, Any]]:
    """Fit whole evidence records into the reader's row-level reference budget."""
    output: list[dict[str, Any]] = []
    used = 0
    for passage in passages:
        rendered = f"- {passage.get('title', '')}: {passage.get('text', '')}"
        cost = len(rendered) + int(bool(output))
        if used + cost > max_chars:
            continue
        output.append(passage)
        used += cost
    return output


def build_references(
    rows: list[dict[str, Any]],
    *,
    evidence_per_claim: int,
    include_nei: bool,
    universe: list[dict[str, Any]] | None = None,
    include_verifier_relation: bool = False,
    max_source_chars: int = 1200,
    max_reference_chars: int = 5000,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("condition")) != "real":
            continue
        if not include_nei and row.get("verdict") == "NOT_ENOUGH_INFO":
            continue
        for evidence in (row.get("evidence") or [])[:evidence_per_claim]:
            sentence = truncate_chars(
                str(evidence.get("text") or "").strip(), max_source_chars
            )
            if not sentence:
                continue
            relation = str(row.get("verdict") or "NOT_ENOUGH_INFO")
            confidence = float(row.get("confidence") or 0.0)
            proposition = str(row.get("proposition") or "").strip()
            relation_line = (
                f"Candidate relation: {relation} (confidence {confidence:.3f})\n"
                if include_verifier_relation else ""
            )
            text = (
                f"Claim: {proposition}\n"
                f"{relation_line}"
                f"Source sentence: {sentence}\n"
                "Instruction: independently check entity identity, temporal compatibility, "
                "and whether the sentence actually settles the claim; ignore it otherwise."
            )
            grouped[(str(row["dataset"]), row["index"])].append({
                "title": str(evidence.get("title") or "Wikipedia"),
                "text": text,
                "url": str(evidence.get("url") or ""),
                "claim_index": int(row["claim_index"]),
            })
    grouped = {
        key: limit_reference_passages(passages, max_reference_chars)
        for key, passages in grouped.items()
    }
    grouped = {key: passages for key, passages in grouped.items() if passages}
    active_keys = sorted(grouped)
    keys = (
        sorted({(str(row["dataset"]), row["index"]) for row in universe})
        if universe is not None
        else active_keys
    )
    donors: dict[tuple[str, Any], tuple[str, Any]] = {}
    for position, row_key in enumerate(active_keys):
        for offset in range(len(active_keys)):
            candidate = active_keys[
                (position + len(active_keys) // 2 + offset) % len(active_keys)
            ]
            if candidate != row_key and candidate[0] != row_key[0]:
                donors[row_key] = candidate
                break
        if row_key not in donors:
            raise ValueError(f"no cross-dataset shuffled donor for {row_key}")
    output = []
    for dataset, index in keys:
        row_key = (dataset, index)
        passages = grouped.get(row_key, [])
        donor = donors.get(row_key)
        output.append({
            "dataset": dataset,
            "index": index,
            "passages": passages,
            "real_passages": passages,
            "shuffled_passages": grouped[donor] if donor is not None else [],
            "shuffled_donor_dataset": donor[0] if donor is not None else None,
            "shuffled_donor_index": donor[1] if donor is not None else None,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-per-claim", type=int, default=1)
    parser.add_argument("--include-nei", action="store_true")
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=1200,
        help="Maximum characters retained from one source sentence.",
    )
    parser.add_argument(
        "--max-reference-chars",
        type=int,
        default=5000,
        help="Maximum rendered evidence characters retained for one response row.",
    )
    parser.add_argument(
        "--row-universe",
        type=Path,
        help="Optional JSONL defining every dataset/index row; uncovered rows get empty evidence.",
    )
    parser.add_argument(
        "--include-verifier-relation",
        action="store_true",
        help="Expose the noisy NLI relation to the teacher; omitted by default to avoid anchoring.",
    )
    args = parser.parse_args()
    output = build_references(
        read_jsonl(args.verification),
        evidence_per_claim=args.evidence_per_claim,
        include_nei=args.include_nei,
        universe=read_jsonl(args.row_universe) if args.row_universe else None,
        include_verifier_relation=args.include_verifier_relation,
        max_source_chars=args.max_source_chars,
        max_reference_chars=args.max_reference_chars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(json.dumps({
        "rows": len(output),
        "passages": sum(len(row["passages"]) for row in output),
        "max_reference_chars": max(
            (
                len("\n".join(
                    f"- {passage.get('title', '')}: {passage.get('text', '')}"
                    for passage in row["passages"]
                ))
                for row in output
            ),
            default=0,
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
