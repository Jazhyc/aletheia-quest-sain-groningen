#!/usr/bin/env python3
"""Audit one black-box judge run from its result.json.

The report is intentionally text/Markdown so it can be pasted into the prompt
optimization docs after each run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("balanced_accuracy", "auroc", "recall", "fpr")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_predictions(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["index"]): row for row in csv.DictReader(handle)}


def balanced_accuracy(labels: list[int], preds: list[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    tp = sum(1 for y, p in zip(labels, preds, strict=True) if y == 1 and p)
    fp = sum(1 for y, p in zip(labels, preds, strict=True) if y == 0 and p)
    recall = tp / positives
    fpr = fp / negatives
    return 0.5 * (recall + (1.0 - fpr))


def sweep_thresholds(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    datasets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        datasets[row["dataset"]].append(row)

    out = []
    for threshold in [i / 100 for i in range(0, 101)]:
        dataset_scores = []
        for drows in datasets.values():
            labels = [int(row["label"]) for row in drows]
            preds = [float(row["score"]) >= threshold for row in drows]
            score = balanced_accuracy(labels, preds)
            if score is not None:
                dataset_scores.append(score)
        if dataset_scores:
            out.append((threshold, mean(dataset_scores)))
    return sorted(out, key=lambda item: item[1], reverse=True)


def trim(text: str, max_chars: int) -> str:
    squashed = " ".join(text.split())
    if len(squashed) <= max_chars:
        return squashed
    return squashed[: max_chars - 3] + "..."


def build_rows(result: dict[str, Any], result_path: Path) -> list[dict[str, Any]]:
    base = result_path.parent
    generations_path = result.get("generations_path")
    if generations_path:
        gen_path = Path(generations_path)
        if not gen_path.is_absolute():
            gen_path = base / gen_path
    else:
        gen_path = None
    generations = load_jsonl(gen_path)
    gen_by_key = {(row["dataset"], int(row["index"])): row for row in generations}

    rows = []
    for dataset in result["datasets"]:
        pred_path = Path(dataset["predictions_path"])
        if not pred_path.is_absolute():
            pred_path = base / pred_path
        predictions = load_predictions(pred_path)
        for index, pred in predictions.items():
            gen = gen_by_key.get((dataset["dataset"], index), {})
            rows.append(
                {
                    "dataset": dataset["dataset"],
                    "index": index,
                    "label": int(gen.get("label", -1)),
                    "score": float(pred["score"]),
                    "deceptive": pred["deceptive"].lower() == "true",
                    "rating": gen.get("rating"),
                    "parse_error": bool(gen.get("parse_error", False)),
                    "text": gen.get("text", ""),
                }
            )
    return rows


def print_report(result_path: Path, max_cases: int) -> None:
    result = load_json(result_path)
    rows = build_rows(result, result_path)
    timing = result.get("timing", {})
    metrics = result.get("metrics", {})

    print(f"# Audit: `{result.get('method')}` / `{result.get('split')}`")
    print()
    print(f"- rows: {result.get('n')}")
    print(f"- score_seconds: {timing.get('score_seconds')}")
    print(f"- rows_per_second: {timing.get('rows_per_second')}")
    print(f"- parse_errors: {result.get('parse_errors')}")
    for key in METRICS:
        print(f"- {key}: {metrics.get(key)}")
    print()

    print("## Best Thresholds")
    print()
    print("| threshold | mean balanced accuracy |")
    print("| ---: | ---: |")
    for threshold, score in sweep_thresholds([row for row in rows if row["label"] in (0, 1)])[:10]:
        print(f"| {threshold:.2f} | {score:.4f} |")
    print()

    print("## Worst Datasets")
    print()
    print("| dataset | bal_acc | AUROC | recall | FPR | n |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    ordered = sorted(result["datasets"], key=lambda item: item["metrics"]["balanced_accuracy"])
    for item in ordered[:8]:
        m = item["metrics"]
        print(
            f"| `{item['dataset']}` | {m['balanced_accuracy']:.4f} | "
            f"{m['auroc']:.4f} | {m['recall']:.4f} | {m['fpr']:.4f} | {item['n']} |"
        )
    print()

    false_negatives = [
        row for row in rows if row["label"] == 1 and not row["deceptive"]
    ]
    false_positives = [
        row for row in rows if row["label"] == 0 and row["deceptive"]
    ]
    parse_errors = [row for row in rows if row["parse_error"]]

    print("## Error Counts")
    print()
    print(f"- false_negatives: {len(false_negatives)}")
    print(f"- false_positives: {len(false_positives)}")
    print(f"- parse_errors: {len(parse_errors)}")
    print()

    def print_cases(title: str, cases: list[dict[str, Any]]) -> None:
        print(f"## {title}")
        print()
        for row in cases[:max_cases]:
            print(f"- `{row['dataset']}` index `{row['index']}`")
            print(f"  label={row['label']} score={row['score']:.4f} rating={row['rating']}")
            print(f"  text: {trim(row['text'], 420)}")
        if not cases:
            print("- none")
        print()

    print_cases("Lowest-Score False Negatives", sorted(false_negatives, key=lambda row: row["score"]))
    print_cases("Highest-Score False Positives", sorted(false_positives, key=lambda row: row["score"], reverse=True))
    print_cases("Parse Errors", parse_errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()
    print_report(args.result_json, args.max_cases)


if __name__ == "__main__":
    main()

