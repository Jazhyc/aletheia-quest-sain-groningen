#!/usr/bin/env python3
"""Add paired row-bootstrap uncertainty to the Q397 OOD transfer result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/blackbox/q397_reasoning_ood_transfer_ndif_v1/scores.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/blackbox/q397_reasoning_ood_transfer_ndif_v1/analysis.json"
)
DEFAULT_STRUCTURAL_INPUT = (
    ROOT
    / "results/blackbox/q397_reasoning_ood_transfer_ndif_v1/"
    "structural_direct_scores.jsonl"
)
DEFAULT_STRUCTURAL_REASONING_INPUT = (
    ROOT
    / "results/blackbox/q397_reasoning_ood_transfer_ndif_v1/"
    "structural_reasoning_scores.jsonl"
)
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20260729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--structural-input",
        type=Path,
        default=DEFAULT_STRUCTURAL_INPUT,
    )
    parser.add_argument(
        "--structural-reasoning-input",
        type=Path,
        default=DEFAULT_STRUCTURAL_REASONING_INPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def paired_bootstrap(
    frame: pd.DataFrame,
    candidate_column: str,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}
    point_deltas: dict[str, float] = {}
    for category, group in frame.groupby("category", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        if len(np.unique(labels)) < 2:
            continue
        direct = group["direct_score"].to_numpy(dtype=float)
        candidate = group[candidate_column].to_numpy(dtype=float)
        negative = np.flatnonzero(labels == 0)
        positive = np.flatnonzero(labels == 1)
        point_deltas[str(category)] = float(
            roc_auc_score(labels, candidate)
            - roc_auc_score(labels, direct)
        )
        negative_draws = rng.choice(
            negative,
            size=(samples, len(negative)),
            replace=True,
        )
        positive_draws = rng.choice(
            positive,
            size=(samples, len(positive)),
            replace=True,
        )

        def bootstrap_aurocs(scores: np.ndarray) -> np.ndarray:
            sampled_negative = scores[negative_draws]
            sampled_positive = scores[positive_draws]
            greater = (
                sampled_positive[:, :, None]
                > sampled_negative[:, None, :]
            )
            equal = (
                sampled_positive[:, :, None]
                == sampled_negative[:, None, :]
            )
            return greater.mean(axis=(1, 2)) + 0.5 * equal.mean(axis=(1, 2))

        category_draws = (
            bootstrap_aurocs(candidate) - bootstrap_aurocs(direct)
        )
        draws[str(category)] = category_draws

    stacked = np.stack(list(draws.values()))
    macro_draws = stacked.mean(axis=0)
    category_rng = np.random.default_rng(seed + 1)
    point_values = np.asarray(list(point_deltas.values()), dtype=np.float64)
    category_draws = category_rng.choice(
        point_values,
        size=(samples, len(point_values)),
        replace=True,
    ).mean(axis=1)
    return {
        "point_macro_delta": float(np.mean(list(point_deltas.values()))),
        "macro_95_interval": [
            float(np.quantile(macro_draws, 0.025)),
            float(np.quantile(macro_draws, 0.975)),
        ],
        "macro_probability_positive": float(np.mean(macro_draws > 0)),
        "category_resampling_95_interval": [
            float(np.quantile(category_draws, 0.025)),
            float(np.quantile(category_draws, 0.975)),
        ],
        "category_resampling_probability_positive": float(
            np.mean(category_draws > 0)
        ),
        "categories": {
            category: {
                "point_delta": point_deltas[category],
                "interval_95": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
                "probability_positive": float(np.mean(values > 0)),
            }
            for category, values in draws.items()
        },
    }


def log_odds_blend(
    direct: pd.Series,
    reasoning: pd.Series,
) -> np.ndarray:
    direct_values = np.clip(
        direct.to_numpy(dtype=float),
        1e-8,
        1 - 1e-8,
    )
    reasoning_values = np.clip(
        reasoning.to_numpy(dtype=float),
        1e-8,
        1 - 1e-8,
    )
    margin = (
        0.6 * (np.log(direct_values) - np.log1p(-direct_values))
        + 0.4 * (
            np.log(reasoning_values) - np.log1p(-reasoning_values)
        )
    )
    return 1.0 / (1.0 + np.exp(-margin))


def main() -> None:
    args = parse_args()
    frame = pd.read_json(args.input, lines=True)
    structural = pd.read_json(args.structural_input, lines=True)
    structural_reasoning = pd.read_json(
        args.structural_reasoning_input,
        lines=True,
    )
    keys = ["category", "index", "source_model"]
    frame = frame.merge(
        structural[keys + ["structural_direct_score"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if frame["structural_direct_score"].isna().any():
        raise RuntimeError("structural direct scores do not cover the OOD sample")
    frame = frame.merge(
        structural_reasoning[
            keys
            + ["structural_reasoning_score", "structural_reasoning_reply"]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if frame["structural_reasoning_score"].isna().any():
        raise RuntimeError("structural reasoning scores do not cover the sample")
    original_direct = frame["direct_score"].copy()
    frame["direct_score"] = frame["structural_direct_score"]
    frame["structural_blend_score"] = log_odds_blend(
        frame["direct_score"],
        frame["reasoning_score"],
    )
    frame["matched_structural_blend_score"] = log_odds_blend(
        frame["direct_score"],
        frame["structural_reasoning_score"],
    )
    analysis = {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tail_prompt_pair": {
            "reasoning_vs_direct": paired_bootstrap(
                frame.assign(direct_score=original_direct),
                "reasoning_score",
            ),
            "blend_vs_direct": paired_bootstrap(
                frame.assign(direct_score=original_direct),
                "blend_score",
            ),
        },
        "current_structural_direct": {
            "tail_reasoning_vs_direct": paired_bootstrap(
                frame,
                "reasoning_score",
            ),
            "tail_reasoning_blend_vs_direct": paired_bootstrap(
                frame,
                "structural_blend_score",
            ),
            "matched_reasoning_vs_direct": paired_bootstrap(
                frame,
                "structural_reasoning_score",
            ),
            "matched_blend_vs_direct": paired_bootstrap(
                frame,
                "matched_structural_blend_score",
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
