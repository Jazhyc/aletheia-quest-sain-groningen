#!/usr/bin/env python3
"""Analyze the frozen 60% Q397 / 40% base-Qwen log-odds blend."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
Q397_WEIGHT = 0.60
BASE_WEIGHT = 0.40
DEFAULT_Q397_VALIDATION = (
    ROOT / "results/blackbox/q397_auxiliary_readout_validation_v1/scores.jsonl"
)
DEFAULT_BASE_VALIDATION = (
    ROOT
    / "results/blackbox/phoenix_base_qwen9b_verbalizer_sweep_validation_v1/"
    "digits_explicit/generations.jsonl"
)
DEFAULT_Q397_TEST = (
    ROOT
    / "results/blackbox/qwen9b_qwen397_tvg_soft_r16_lr5e5_ep2_v1/"
    "test_vllm_logits/prediction/predictions.csv"
)
DEFAULT_BASE_TEST = (
    ROOT / "results/blackbox/q397_base_explicit_ensemble_test_base_v1/scores.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/blackbox/q397_base_explicit_ensemble_v1"


def logit(probability: float) -> float:
    """Convert a bounded probability to log odds."""
    value = min(max(float(probability), 1.0e-12), 1.0 - 1.0e-12)
    return math.log(value) - math.log1p(-value)


def sigmoid(value: float) -> float:
    """Return a numerically stable probability."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


def load_jsonl(path: Path, score_path: tuple[str, ...]) -> dict[tuple[str, str], dict[str, Any]]:
    """Load keyed JSONL rows and normalize a nested score field."""
    result = {}
    for row in map(json.loads, path.read_text().splitlines()):
        value: Any = row
        for key in score_path:
            value = value[key]
        row = dict(row)
        row["_score"] = float(value)
        result[(str(row["dataset"]), str(row["index"]))] = row
    return result


def load_csv(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load keyed CSV rows."""
    with path.open(newline="") as handle:
        return {
            (str(row["dataset"]), str(row["index"])): {
                **row,
                "label": int(row["label"]),
                "_score": float(row["score"]),
            }
            for row in csv.DictReader(handle)
        }


def per_dataset(
    rows: list[dict[str, Any]], scores: list[float]
) -> dict[str, float]:
    """Compute AUROC for each dataset unit."""
    groups: dict[str, tuple[list[int], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for row, score in zip(rows, scores, strict=True):
        labels, values = groups[str(row["dataset"])]
        labels.append(int(row["label"]))
        values.append(float(score))
    return {
        dataset: float(roc_auc_score(labels, values))
        for dataset, (labels, values) in groups.items()
    }


def summarize(
    rows: list[dict[str, Any]],
    baseline: list[float],
    candidate: list[float],
) -> dict[str, Any]:
    """Summarize the paired blend against Q397."""
    base_units = per_dataset(rows, baseline)
    candidate_units = per_dataset(rows, candidate)
    deltas = {
        dataset: candidate_units[dataset] - base_units[dataset]
        for dataset in base_units
    }
    scenario = {}
    for name in ("instructed-deception", "varied-deception"):
        datasets = [dataset for dataset in base_units if name in dataset]
        scenario[name] = {
            "baseline": float(np.mean([base_units[item] for item in datasets])),
            "candidate": float(
                np.mean([candidate_units[item] for item in datasets])
            ),
        }
        scenario[name]["delta"] = (
            scenario[name]["candidate"] - scenario[name]["baseline"]
        )
    return {
        "baseline_macro_auroc": float(np.mean(list(base_units.values()))),
        "candidate_macro_auroc": float(np.mean(list(candidate_units.values()))),
        "delta": float(np.mean(list(deltas.values()))),
        "scenario": scenario,
        "unit_wins_ties_losses": {
            "wins": sum(value > 1.0e-12 for value in deltas.values()),
            "ties": sum(abs(value) <= 1.0e-12 for value in deltas.values()),
            "losses": sum(value < -1.0e-12 for value in deltas.values()),
        },
        "worst_unit_delta": min(deltas.values()),
        "unique_scores": {
            "baseline": int(np.unique(baseline).size),
            "candidate": int(np.unique(candidate).size),
        },
        "per_dataset": {
            dataset: {
                "baseline": base_units[dataset],
                "candidate": candidate_units[dataset],
                "delta": deltas[dataset],
            }
            for dataset in sorted(base_units)
        },
    }


def paired_rows(
    q397: dict[tuple[str, str], dict[str, Any]],
    base: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    """Align two streams and apply the frozen log-odds blend."""
    if q397.keys() != base.keys():
        raise ValueError(
            f"score keys differ: q397={len(q397)} base={len(base)} "
            f"matched={len(q397.keys() & base.keys())}"
        )
    keys = sorted(q397)
    rows = [q397[key] for key in keys]
    q397_scores = [q397[key]["_score"] for key in keys]
    candidate = [
        sigmoid(
            Q397_WEIGHT * logit(q397[key]["_score"])
            + BASE_WEIGHT * logit(base[key]["_score"])
        )
        for key in keys
    ]
    return rows, q397_scores, candidate


def analyze_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Reproduce the frozen validation selection."""
    q397 = load_jsonl(args.q397_validation, ("scores", "digits"))
    base = load_jsonl(args.base_validation, ("score",))
    return summarize(*paired_rows(q397, base))


def analyze_test(args: argparse.Namespace) -> dict[str, Any] | None:
    """Evaluate test only when the new base stream exists."""
    if not args.base_test.exists():
        return None
    q397 = load_csv(args.q397_test)
    base = load_jsonl(args.base_test, ("score",))
    return summarize(*paired_rows(q397, base))


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Write the validation analysis and optional frozen test confirmation."""
    report = {
        "method": "q397_base_explicit_ensemble_v1",
        "blend": {
            "q397_weight": Q397_WEIGHT,
            "base_explicit_weight": BASE_WEIGHT,
            "space": "log_odds",
            "reasoning_required": False,
        },
        "validation": analyze_validation(args),
        "test": analyze_test(args),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q397-validation", type=Path, default=DEFAULT_Q397_VALIDATION
    )
    parser.add_argument(
        "--base-validation", type=Path, default=DEFAULT_BASE_VALIDATION
    )
    parser.add_argument("--q397-test", type=Path, default=DEFAULT_Q397_TEST)
    parser.add_argument("--base-test", type=Path, default=DEFAULT_BASE_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
