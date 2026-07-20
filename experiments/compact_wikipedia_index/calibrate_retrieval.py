#!/usr/bin/env python3
"""Calibrate conservative lexical emission on train-only audited references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compact_wikipedia_index.core import retrieve, source_matches
from experiments.fever_fact_verification.core import load_grounded_claims


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reference_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, Any, int], list[dict[str, Any]]]:
    output: dict[tuple[str, Any, int], list[dict[str, Any]]] = {}
    for row in rows:
        for passage in row.get("real_passages") or row.get("passages") or []:
            if passage.get("claim_index") is None:
                continue
            key = (str(row["dataset"]), row["index"], int(passage["claim_index"]))
            output.setdefault(key, []).append(passage)
    return output


def threshold_report(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    emitted = [row for row in rows if float(row["score"]) >= threshold]
    matches = sum(bool(row["strict_match"]) for row in emitted)
    accepted = sum(bool(row["audited_reference_exists"]) for row in emitted)
    all_matches = sum(bool(row["strict_match"]) for row in rows)
    return {
        "threshold": threshold,
        "emitted": len(emitted),
        "strict_matches": matches,
        "strict_precision": matches / len(emitted) if emitted else 0.0,
        "audited_claim_rate": accepted / len(emitted) if emitted else 0.0,
        "strict_match_recall": matches / all_matches if all_matches else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Deterministic claim sample used to keep calibration CPU-cheap.",
    )
    args = parser.parse_args()

    claims = load_grounded_claims(
        read_jsonl(args.claims), dataset_name_contains="varied-deception"
    )
    if args.sample and len(claims) > args.sample:
        claims.sort(
            key=lambda claim: hashlib.sha256(
                f"{claim['dataset']}\0{claim['index']}\0{claim['claim_index']}".encode()
            ).digest()
        )
        claims = claims[: args.sample]
    references = reference_map(read_jsonl(args.references))
    scored = []
    with sqlite3.connect(f"file:{args.index}?mode=ro", uri=True) as connection:
        for claim in claims:
            key = (str(claim["dataset"]), claim["index"], int(claim["claim_index"]))
            query = f"{claim['question']} {claim['quote']}"
            candidates = retrieve(connection, query, limit=2, candidate_limit=30)
            top = candidates[0] if candidates else None
            second_score = (
                float(candidates[1]["retrieval_score"])
                if len(candidates) > 1
                else 0.0
            )
            accepted = references.get(key, [])
            scored.append({
                **{field: claim[field] for field in (
                    "dataset", "index", "claim_index", "label", "quote",
                    "proposition", "question", "teacher_assessment",
                )},
                "score": float(top["retrieval_score"]) if top else 0.0,
                "score_gap": (
                    float(top["retrieval_score"]) - second_score if top else 0.0
                ),
                "audited_reference_exists": bool(accepted),
                "strict_match": bool(
                    top and any(source_matches(top, reference) for reference in accepted)
                ),
                "candidate": top,
            })
    thresholds = sorted({round(float(row["score"]), 3) for row in scored})
    reports = [threshold_report(scored, threshold) for threshold in thresholds]
    eligible = [
        report
        for report in reports
        if report["emitted"] >= 100 and report["audited_claim_rate"] >= 0.70
    ]
    selected = max(
        eligible,
        key=lambda report: (report["emitted"], -report["threshold"]),
        default=None,
    )
    report = {
        "claims": len(scored),
        "audited_claims": sum(row["audited_reference_exists"] for row in scored),
        "strict_top1_matches": sum(row["strict_match"] for row in scored),
        "selected": selected,
        "thresholds": reports,
        "rows": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"thresholds", "rows"}}, indent=2))


if __name__ == "__main__":
    main()
