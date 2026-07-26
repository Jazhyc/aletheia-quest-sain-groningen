#!/usr/bin/env python3
"""Measure semantic-retrieval and label-aware utility oracles for Phoenix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.wikidata_rag.evaluate_qwen_planner_phoenix import (
    summarize_condition,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def best_for_label(
    scores: list[float],
    label: int,
) -> tuple[float, int]:
    """Return the most label-helpful score and its position."""
    if not scores:
        raise ValueError("oracle needs at least the no-evidence score")
    index = max(range(len(scores)), key=scores.__getitem__) if label else min(
        range(len(scores)), key=scores.__getitem__
    )
    return float(scores[index]), index


def row_oracle_scores(row: dict[str, Any]) -> dict[str, float | int | bool]:
    """Construct progressively stronger candidate-selection oracles for one row."""
    empty = float(row["empty_score"])
    label = int(row["label"])
    candidates = row.get("candidates", [])
    decisive = [
        candidate
        for candidate in candidates
        if candidate.get("semantic_label") == "decisive"
    ]

    # This is a retrieval-label oracle, not a deception-label oracle: use the
    # first teacher-decisive fact and accept whatever effect it has on Phoenix.
    semantic_forced = float(decisive[0]["score"]) if decisive else empty

    semantic_scores = [empty, *[float(item["score"]) for item in decisive]]
    semantic_utility, semantic_choice = best_for_label(semantic_scores, label)
    all_scores = [empty, *[float(item["score"]) for item in candidates]]
    all_utility, all_choice = best_for_label(all_scores, label)
    return {
        "recomputed_empty": empty,
        "semantic_retrieval_oracle": semantic_forced,
        "semantic_utility_oracle": semantic_utility,
        "any_candidate_utility_oracle": all_utility,
        "has_decisive": bool(decisive),
        "semantic_utility_used_fact": semantic_choice > 0,
        "any_utility_used_fact": all_choice > 0,
    }


def compose_oracles(
    baseline: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Replace candidate-cache rows while preserving all other baseline scores."""
    keys = ["dataset", "index"]
    base = baseline.set_index(keys).copy()
    updates = []
    for row in rows:
        key = (str(row["dataset"]), row["index"])
        if key not in base.index:
            raise KeyError(f"candidate row absent from baseline: {key}")
        if int(row["label"]) != int(base.loc[key, "label"]):
            raise ValueError(f"label mismatch for candidate row: {key}")
        updates.append({**{"dataset": key[0], "index": key[1]}, **row_oracle_scores(row)})
    update = pd.DataFrame(updates).set_index(keys)
    conditions = {"baseline": base.reset_index()}
    for name in (
        "recomputed_empty",
        "semantic_retrieval_oracle",
        "semantic_utility_oracle",
        "any_candidate_utility_oracle",
    ):
        frame = base.copy()
        frame.loc[update.index, "score"] = update[name]
        conditions[name] = frame.reset_index()
    counts = {
        "candidate_cache_rows": len(update),
        "rows_with_decisive": int(update["has_decisive"].sum()),
        "semantic_utility_used_fact": int(update["semantic_utility_used_fact"].sum()),
        "any_utility_used_fact": int(update["any_utility_used_fact"].sum()),
    }
    return conditions, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--baseline-generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.candidate_scores)
    baseline = pd.read_json(args.baseline_generations, lines=True)[
        ["dataset", "index", "label", "direct_margin_score"]
    ].rename(columns={"direct_margin_score": "score"})
    conditions, counts = compose_oracles(baseline, rows)
    summaries = {
        name: summarize_condition(frame) for name, frame in conditions.items()
    }
    reference = summaries["recomputed_empty"]["metrics"]
    result = {
        **counts,
        "conditions": summaries,
        "uplift_vs_recomputed_empty": {
            name: {
                scenario: (
                    report["metrics"][scenario]["auroc"]
                    - reference[scenario]["auroc"]
                )
                for scenario in ("all", "instructed", "varied")
            }
            for name, report in summaries.items()
            if name not in {"baseline", "recomputed_empty"}
        },
        "candidate_scores": args.candidate_scores.resolve().as_posix(),
        "baseline_generations": args.baseline_generations.resolve().as_posix(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
