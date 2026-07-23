#!/usr/bin/env python3
"""Apply a utility ranker and build matched empty/real/shuffled judge conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.wikidata_rag.train_utility_retriever import (
    semantic_scores,
    utility_feature_dict,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_model(path: Path) -> dict[str, Any]:
    artifact = np.load(path, allow_pickle=False)
    semantic_model = None
    if "semantic_weights" in artifact:
        semantic_model = {
            "weights": artifact["semantic_weights"].astype(np.float32).reshape(-1),
            "intercept": float(artifact["semantic_intercept"].reshape(-1)[0]),
        }
    return {
        "weights": artifact["weights"].astype(np.float32).reshape(-1),
        "intercept": float(artifact["intercept"].reshape(-1)[0]),
        "threshold": float(artifact["threshold"].reshape(-1)[0]),
        "feature_mode": str(artifact["feature_mode"].reshape(-1)[0]),
        "semantic_top_k": (
            int(artifact["semantic_top_k"].reshape(-1)[0])
            if "semantic_top_k" in artifact else -1
        ),
        "semantic_model": semantic_model,
    }


def select_candidate(row: dict[str, Any], model: dict[str, Any]) -> dict[str, Any] | None:
    semantic = semantic_scores(row, model["semantic_model"])
    ranks = np.argsort(np.argsort(-np.asarray(semantic))).tolist() if semantic else []
    allowed = set(range(len(row["candidates"])))
    if model["semantic_top_k"] > 0:
        allowed = set(
            int(index)
            for index in np.argsort(-np.asarray(semantic))[: model["semantic_top_k"]]
        )
    scores = []
    for candidate_index, (candidate, semantic_score, rank) in enumerate(zip(
        row["candidates"], semantic, ranks, strict=True
    )):
        if candidate_index not in allowed:
            scores.append(-float("inf"))
            continue
        features = utility_feature_dict(
            row, candidate, feature_mode=model["feature_mode"],
            semantic_score=semantic_score, semantic_rank=int(rank),
        )
        score = model["intercept"] + sum(
            model["weights"][index] * value for index, value in features.items()
        )
        scores.append(float(score))
    if not scores:
        return None
    best = int(np.argmax(scores))
    if scores[best] < model["threshold"]:
        return None
    return dict(row["candidates"][best]) | {"utility_ranker_score": scores[best]}


def deranged_donors(selected: list[dict[str, Any] | None]) -> list[dict[str, Any] | None]:
    active = [index for index, candidate in enumerate(selected) if candidate is not None]
    donors: list[dict[str, Any] | None] = [None] * len(selected)
    if len(active) < 2:
        return donors
    rotated = active[1:] + active[:1]
    for recipient, donor in zip(active, rotated, strict=True):
        donors[recipient] = selected[donor]
    return donors


def passage(candidate: dict[str, Any] | None) -> list[dict[str, str]]:
    if candidate is None:
        return []
    return [{"title": str(candidate["subject"]), "text": str(candidate["fact"])}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    model = load_model(args.model)
    selected = [select_candidate(row, model) for row in rows]
    shuffled = deranged_donors(selected)
    output = []
    for row, real, noise in zip(rows, selected, shuffled, strict=True):
        output.append({
            "dataset": row["dataset"], "index": row["index"],
            "real_passages": passage(real),
            "shuffled_passages": passage(noise),
            "selected_candidate": real,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(
        f"wrote {len(output)} rows with {sum(value is not None for value in selected)} "
        f"utility-gated facts to {args.output}"
    )


if __name__ == "__main__":
    main()
