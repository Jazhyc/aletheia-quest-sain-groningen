#!/usr/bin/env python3
"""Learn tiny high-precision question-phrase to Wikidata-relation rules."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from experiments.wikidata_rag.claim_retrieval import normalize


DECISIVE = {"decisive", "supports", "contradicts"}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bucket(group: str) -> int:
    return hashlib.sha1(group.encode()).digest()[0] % 5


def phrase_features(question: str) -> set[str]:
    # Function words such as "who", "by", and "which" are often the most
    # informative relation cues, so do not apply the retrieval stopword list.
    words = normalize(question).split()[:40]
    features = {f"w={word}" for word in words}
    for size in (2, 3):
        features.update(
            f"g{size}={'|'.join(words[start:start + size])}"
            for start in range(len(words) - size + 1)
        )
    first = normalize(question).partition(" ")[0]
    if first:
        features.add(f"first={first}")
    return features


def group_observations(rows: Iterable[dict[str, Any]]) -> list[tuple[str, str, bool]]:
    """Deduplicate repeated answer variants at question-group/predicate level."""
    grouped: dict[tuple[str, str], tuple[str, bool]] = {}
    for row in rows:
        candidates = {item["id"]: item for item in row["candidates"]}
        decisive_predicates = {
            candidates[item["id"]]["predicate"]
            for item in row["labels"]
            if item["label"] in DECISIVE and item["id"] in candidates
        }
        for predicate in {item["predicate"] for item in row["candidates"]}:
            key = (row["question_group"], predicate)
            old = grouped.get(key)
            grouped[key] = (row["question"], predicate in decisive_predicates or bool(old and old[1]))
    return [(question, predicate, positive) for (_, predicate), (question, positive) in grouped.items()]


def wilson_lower(positive: int, total: int, z: float = 1.0) -> float:
    if total == 0:
        return 0.0
    rate = positive / total
    denominator = 1.0 + z * z / total
    centre = rate + z * z / (2.0 * total)
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (centre - margin) / denominator


def learn_rules(
    rows: Iterable[dict[str, Any]], minimum_support: int
) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for question, predicate, positive in group_observations(rows):
        for feature in phrase_features(question):
            counts[(predicate, feature)][0] += int(positive)
            counts[(predicate, feature)][1] += 1
    return {
        key: wilson_lower(positive, total)
        for key, (positive, total) in counts.items()
        if total >= minimum_support and positive > 0
    }


def candidate_score(row: dict[str, Any], candidate: dict[str, Any], rules: dict[tuple[str, str], float]) -> float:
    predicate = candidate["predicate"]
    return max(
        (rules.get((predicate, feature), 0.0) for feature in phrase_features(row["question"])),
        default=0.0,
    )


def selections(
    rows: list[dict[str, Any]], rules: dict[tuple[str, str], float]
) -> list[tuple[float, bool, bool]]:
    output = []
    for row in rows:
        decisive_ids = {
            item["id"] for item in row["labels"] if item["label"] in DECISIVE
        }
        scored = [(candidate_score(row, candidate, rules), candidate) for candidate in row["candidates"]]
        score, candidate = max(scored, key=lambda item: item[0], default=(0.0, None))
        output.append((
            score, bool(candidate and candidate["id"] in decisive_ids), bool(decisive_ids)
        ))
    return output


def operating_point(selected: list[tuple[float, bool, bool]], threshold: float) -> dict[str, float]:
    emitted = sum(score >= threshold for score, _, _ in selected)
    correct = sum(score >= threshold and positive for score, positive, _ in selected)
    rows_with_decisive = sum(has_decisive for _, _, has_decisive in selected)
    return {
        "rows": len(selected), "rows_with_decisive": rows_with_decisive,
        "emitted": emitted, "decisive": correct,
        "precision": correct / max(1, emitted),
        "selected_decisive_recall": correct / max(1, rows_with_decisive),
    }


def choose_threshold(selected: list[tuple[float, bool, bool]], minimum_precision: float) -> tuple[float, dict[str, float]]:
    candidates = []
    for threshold in sorted({score for score, _, _ in selected if score > 0.0}):
        report = operating_point(selected, threshold)
        if report["emitted"] >= 5 and report["precision"] >= minimum_precision:
            candidates.append((report["decisive"], report["emitted"], threshold, report))
    if not candidates:
        threshold = max((score for score, _, _ in selected), default=0.0) + 1e-6
        return threshold, operating_point(selected, threshold)
    _, _, threshold, report = max(candidates)
    return threshold, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--output-rules", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.8)
    args = parser.parse_args()

    rows = [row for row in load(args.input) if not row.get("parse_error")]
    validation = [row for row in load(args.validation_input) if not row.get("parse_error")]
    development = [row for row in rows if bucket(row["question_group"]) not in {0, 1}]
    calibration = [row for row in rows if bucket(row["question_group"]) == 1]
    internal_test = [row for row in rows if bucket(row["question_group"]) == 0]
    sweep = []
    learned = []
    for minimum_support in (2, 3, 5, 8, 12):
        rules = learn_rules(development, minimum_support)
        threshold, calibration_report = choose_threshold(
            selections(calibration, rules), args.minimum_precision
        )
        sweep.append({
            "minimum_support": minimum_support, "rules": len(rules),
            "threshold": threshold, "calibration": calibration_report,
            "internal_test": operating_point(selections(internal_test, rules), threshold),
            "frozen_validation": operating_point(selections(validation, rules), threshold),
        })
        learned.append(rules)
    selected_index = max(range(len(sweep)), key=lambda index: (
        sweep[index]["calibration"]["decisive"],
        sweep[index]["calibration"]["precision"],
        -sweep[index]["rules"],
    ))
    selected = sweep[selected_index]
    rules = learned[selected_index]
    serialized = [
        {"predicate": predicate, "feature": feature, "score": score}
        for (predicate, feature), score in sorted(rules.items())
        if score >= selected["threshold"]
    ]
    args.output_rules.parent.mkdir(parents=True, exist_ok=True)
    args.output_rules.write_text(json.dumps({
        "threshold": selected["threshold"], "rules": serialized,
    }, separators=(",", ":")) + "\n")
    report = {
        "train_rows": len(rows), "validation_rows": len(validation),
        "development_rows": len(development), "calibration_rows": len(calibration),
        "internal_test_rows": len(internal_test), "sweep": sweep,
        "selected": selected, "serialized_rules": len(serialized),
        "artifact_bytes": args.output_rules.stat().st_size,
    }
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
