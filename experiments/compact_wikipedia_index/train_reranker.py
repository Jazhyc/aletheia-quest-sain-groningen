#!/usr/bin/env python3
"""Fit a tiny proposition-aware reranker on label-blind training audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compact_wikipedia_index.core import (
    FEATURE_NAMES,
    retrieve_proposition,
    source_matches,
)


DECISIVE = {"DECISIVE_SUPPORT", "DECISIVE_CONTRADICTION"}


def stable_bucket(question: str, buckets: int = 5) -> int:
    digest = hashlib.sha256(question.casefold().strip().encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def read_audits(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("parse_valid") or not row.get("proposition"):
                continue
            rows.append(row)
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{row['dataset']}\0{row['index']}\0{row['claim_index']}".encode()
        ).digest()
    )
    return rows[:limit] if limit else rows


def candidate_rows(
    connection: sqlite3.Connection,
    audits: list[dict[str, Any]],
    *,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    output = []
    for position, audit in enumerate(audits, start=1):
        reference = (
            audit.get("selected_candidate")
            if audit.get("decision") in DECISIVE
            else None
        )
        candidates = retrieve_proposition(
            connection,
            str(audit["question"]),
            str(audit["proposition"]),
            limit=candidate_limit,
            candidate_limit=30,
        )
        for candidate in candidates:
            output.append({
                "question": str(audit["question"]),
                "claim_key": (
                    str(audit["dataset"]),
                    audit["index"],
                    int(audit["claim_index"]),
                ),
                "features": candidate["features"],
                "positive": bool(reference and source_matches(candidate, reference)),
                "reference_exists": reference is not None,
                "candidate": candidate,
            })
        if position % 500 == 0:
            print(f"retrieved {position}/{len(audits)} claims", flush=True)
    return output


def claim_report(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    grouped: dict[tuple[str, Any, int], list[tuple[dict[str, Any], float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row["claim_key"], []).append((row, float(score)))
    top = [max(items, key=lambda item: item[1]) for items in grouped.values()]
    emitted = [item for item in top if item[1] >= threshold]
    strict = sum(bool(item[0]["positive"]) for item in emitted)
    decisive = sum(bool(item[0]["reference_exists"]) for item in emitted)
    available = sum(any(row["positive"] for row, _ in items) for items in grouped.values())
    return {
        "claims": len(grouped),
        "source_available": available,
        "emitted": len(emitted),
        "strict_matches": strict,
        "strict_precision": strict / len(emitted) if emitted else 0.0,
        "decisive_claim_rate": decisive / len(emitted) if emitted else 0.0,
        "strict_recall": strict / available if available else 0.0,
    }


def select_threshold(rows: list[dict[str, Any]], scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = sorted({round(float(score), 4) for score in scores}, reverse=True)
    reports = []
    for threshold in candidates:
        report = claim_report(rows, scores, threshold)
        reports.append({"threshold": threshold, **report})
    eligible = [
        report
        for report in reports
        if report["emitted"] >= 20
        and report["strict_precision"] >= 0.25
        and report["decisive_claim_rate"] >= 0.70
    ]
    if eligible:
        selected = max(eligible, key=lambda report: (report["emitted"], report["strict_precision"]))
    else:
        selected = max(reports, key=lambda report: (report["strict_precision"], report["emitted"]))
    return float(selected["threshold"]), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--candidate-limit", type=int, default=16)
    args = parser.parse_args()

    audits = read_audits(args.audits, limit=args.limit)
    with sqlite3.connect(f"file:{args.index}?mode=ro", uri=True) as connection:
        rows = candidate_rows(
            connection,
            audits,
            candidate_limit=args.candidate_limit,
        )
    fit_rows = [row for row in rows if stable_bucket(row["question"]) != 0]
    calibration_rows = [row for row in rows if stable_bucket(row["question"]) == 0]
    if not fit_rows or not calibration_rows:
        raise RuntimeError("question-grouped fit/calibration split is empty")

    x_fit = np.asarray([row["features"] for row in fit_rows], dtype=np.float64)
    y_fit = np.asarray([row["positive"] for row in fit_rows], dtype=np.int64)
    scaler = StandardScaler().fit(x_fit)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ).fit(scaler.transform(x_fit), y_fit)

    fit_scores = classifier.predict_proba(scaler.transform(x_fit))[:, 1]
    x_calibration = np.asarray(
        [row["features"] for row in calibration_rows], dtype=np.float64
    )
    calibration_scores = classifier.predict_proba(
        scaler.transform(x_calibration)
    )[:, 1]
    threshold, selected = select_threshold(calibration_rows, calibration_scores)
    payload = {
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "threshold": threshold,
        "provenance": {
            "audits": str(args.audits),
            "sampled_claims": len(audits),
            "split": "sha256(normalized exact question) modulo 5; bucket 0 calibration",
            "deception_labels_used": False,
        },
    }
    report = {
        "audits": len(audits),
        "candidate_rows": len(rows),
        "fit": claim_report(fit_rows, fit_scores, threshold),
        "calibration": claim_report(calibration_rows, calibration_scores, threshold),
        "fit_all_top1": claim_report(fit_rows, fit_scores, 0.0),
        "calibration_all_top1": claim_report(
            calibration_rows, calibration_scores, 0.0
        ),
        "selected": selected,
        "positive_candidates_fit": int(sum(row["positive"] for row in fit_rows)),
        "positive_candidates_calibration": int(
            sum(row["positive"] for row in calibration_rows)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
