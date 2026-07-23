#!/usr/bin/env python3
"""Apply a compact GPT-OSS-supervised fact ranker to frozen candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.wikidata_rag.train_fact_ranker import matrix
from experiments.wikidata_rag.train_fact_ranker import feature_dict
from experiments.wikidata_rag.train_gptoss_fact_ranker import (
    char_relation_feature_dict,
    supervision_feature_dict,
)


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = np.load(args.model)
    weights = artifact["weights"].astype(np.float32).reshape(-1)
    intercept = float(artifact["intercept"].reshape(-1)[0])
    threshold = float(artifact["threshold"].reshape(-1)[0])
    feature_mode = str(artifact["feature_mode"].reshape(-1)[0]) if "feature_mode" in artifact else "generic"
    confidence_mode = str(artifact["confidence_mode"].reshape(-1)[0]) if "confidence_mode" in artifact else "absolute"
    feature_function = {
        "generic": feature_dict,
        "relation": supervision_feature_dict,
        "char_relation": char_relation_feature_dict,
    }[feature_mode]
    output = []
    for row in load(args.input):
        candidates = row.get("candidates", [])
        if candidates:
            x = matrix([
                feature_function(
                    row["question"], row["answer_full"], set(row["rule_predicates"]),
                    candidate["subject"], candidate["fact"], candidate["popularity"],
                )
                for candidate in candidates
            ])
            scores = np.asarray(x @ weights + intercept).reshape(-1)
            best_index = int(np.argmax(scores))
            best = {**candidates[best_index], "ranker_score": float(scores[best_index])}
            if confidence_mode == "margin" and len(scores) > 1:
                confidence = float(scores[best_index] - np.partition(scores, -2)[-2])
            else:
                confidence = float(scores[best_index])
            best["ranker_confidence"] = confidence
            emitted = bool(confidence >= threshold)
        else:
            best = None
            emitted = False
        output.append({
            "dataset": row["dataset"], "index": row["index"],
            "question": row["question"], "answer_full": row["answer_full"],
            "question_group": row["question_group"], "threshold": threshold,
            "emitted": emitted, "best_candidate": best,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(f"wrote {len(output)} rows; emitted={sum(row['emitted'] for row in output)}")


if __name__ == "__main__":
    main()
