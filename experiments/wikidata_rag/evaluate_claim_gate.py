#!/usr/bin/env python3
"""Compare broad and claim-gated retrieval against frozen teacher summaries."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from experiments.wikidata_rag.evaluate_retrieval import score_record


def load(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def raw_evidence(record: dict[str, object]) -> str:
    if record.get("real_passages"):
        return "\n".join(
            f"{passage['title']}: {passage['text']}"
            for passage in record.get("real_passages", [])
        )
    return "\n".join(
        " | ".join([
            str(entity.get("label", "")), str(entity.get("description", "")),
            "; ".join(entity.get("facts", [])),
        ])
        for entity in record.get("entities", [])
    )


def gated_evidence(record: dict[str, object]) -> str:
    return "\n".join(
        f"{passage['title']}: {passage['text']}"
        for passage in record.get("real_passages", [])
    )


def mean(rows: list[dict[str, float]], key: str) -> float:
    return statistics.fmean(row[key] for row in rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--broad-cache", type=Path, required=True)
    parser.add_argument("--gated-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    teachers = {
        (row["dataset"], row["index"]): row
        for row in load(args.teacher_cache)
        if "varied-deception" in str(row.get("dataset", ""))
        and not row.get("parse_error")
        and row.get("reasoning_summary")
    }
    broad = {(row["dataset"], row["index"]): row for row in load(args.broad_cache)}
    gated = {(row["dataset"], row["index"]): row for row in load(args.gated_cache)}
    keys = sorted(teachers.keys() & broad.keys() & gated.keys())
    scored = []
    for key in keys:
        teacher = teachers[key]
        source = {
            "conversation": teacher["student_prompt"],
            "reasoning_summary": teacher["reasoning_summary"],
        }
        broad_scores = score_record(source, raw_evidence(broad[key]))
        gate_text = gated_evidence(gated[key])
        gate_scores = score_record(source, gate_text)
        scored.append({
            "label": int(teacher["label"]),
            "covered": float(bool(gate_text)),
            **{f"broad_{name}": value for name, value in broad_scores.items()},
            **{f"gated_{name}": value for name, value in gate_scores.items()},
        })

    report: dict[str, object] = {
        "rows": len(scored),
        "abstain_reasons": Counter(
            str(row.get("abstain_reason")) for row in gated.values()
        ),
        "groups": {},
    }
    for label, name in ((None, "all"), (0, "honest"), (1, "deceptive")):
        rows = [row for row in scored if label is None or row["label"] == label]
        covered = [row for row in rows if row["covered"]]
        report["groups"][name] = {
            "rows": len(rows),
            "coverage": len(covered) / max(1, len(rows)),
            "broad_novel_target_recall": mean(rows, "broad_novel_target_recall"),
            "broad_novel_target_recall_covered": mean(covered, "broad_novel_target_recall"),
            "gated_novel_target_recall_all": mean(rows, "gated_novel_target_recall"),
            "gated_novel_target_recall_covered": mean(covered, "gated_novel_target_recall"),
            "broad_evidence_precision": mean(rows, "broad_evidence_precision"),
            "broad_evidence_precision_covered": mean(covered, "broad_evidence_precision"),
            "gated_evidence_precision_covered": mean(covered, "gated_evidence_precision"),
            "covered_with_novel_target_hit": sum(
                row["gated_novel_target_recall"] > 0 for row in covered
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
