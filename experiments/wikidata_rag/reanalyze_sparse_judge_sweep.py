#!/usr/bin/env python3
"""Recompute a sparse evidence sweep while reusing identical empty outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.evaluate_student_sft import (
    metrics_for_score,
)
from experiments.wikidata_rag.evaluate_judge_sweep import (
    CONDITIONS,
    SCORE_COLUMNS,
    active_evidence_keys,
    paired_changes,
    paired_score_deltas,
    reuse_empty_for_inactive_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    evaluated = pd.read_json(args.generations, lines=True)
    active = active_evidence_keys(args.cache)
    evaluated = reuse_empty_for_inactive_rows(evaluated, active)
    conditions = {
        condition: {
            score: metrics_for_score(
                evaluated[evaluated["condition"] == condition], score
            )
            for score in SCORE_COLUMNS
        }
        for condition in CONDITIONS
    }
    comparisons = {}
    for condition in ("real", "shuffled"):
        comparisons[f"{condition}_vs_empty"] = {
            "transitions": paired_changes(evaluated, condition),
            "score_deltas": {
                score: paired_score_deltas(evaluated, condition, score)
                for score in SCORE_COLUMNS
            },
        }
    comparisons["real_vs_shuffled"] = {
        "transitions": paired_changes(evaluated, "real", baseline="shuffled"),
        "score_deltas": {
            score: paired_score_deltas(
                evaluated, "real", score, baseline="shuffled"
            )
            for score in SCORE_COLUMNS
        },
    }
    result = {
        "active_varied_rows": len(active),
        "inactive_outputs_reused": True,
        "conditions": conditions,
        "comparisons": comparisons,
        "parse_errors": {
            condition: int(evaluated.loc[
                evaluated["condition"] == condition, "parse_error"
            ].sum())
            for condition in CONDITIONS
        },
        "cache": args.cache.resolve().as_posix(),
        "source_generations": args.generations.resolve().as_posix(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluated.to_json(
        args.output_dir / "generations.jsonl", orient="records", lines=True
    )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
