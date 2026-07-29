#!/usr/bin/env python3
"""Measure blends and deployable routing between direct and reasoning margins."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "results/blackbox/q397_reasoning_router_validation_v1/scores.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "results/blackbox/q397_reasoning_router_validation_v1/analysis.json"
)
DEFAULT_TEST_INPUT = (
    ROOT / "results/blackbox/q397_reasoning_router_test_v1/scores.jsonl"
)
EPSILON = 1e-8
BLEND_WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
CANDIDATE_CS = (0.01, 0.1, 1.0)
PREDICTION_RE = re.compile(r"Prediction:\s*([01])")


@dataclass(frozen=True)
class Gate:
    name: str
    use_reasoning: Callable[[pd.DataFrame], np.ndarray]


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def macro_auroc(frame: pd.DataFrame, scores: np.ndarray) -> float:
    values = []
    for _, group in frame.assign(_score=scores).groupby("dataset", sort=True):
        values.append(roc_auc_score(group["label"], group["_score"]))
    return float(np.mean(values))


def scenario_aurocs(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    output = {}
    for scenario in ("instructed-deception", "varied-deception"):
        mask = frame["dataset"].str.contains(scenario, regex=False)
        output[scenario] = macro_auroc(
            frame.loc[mask].reset_index(drop=True),
            np.asarray(scores)[mask.to_numpy()],
        )
    return output


def summarize(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    return {
        "macro_auroc": macro_auroc(frame, scores),
        "scenario_auroc": scenario_aurocs(frame, scores),
        "unique_scores": int(np.unique(scores).size),
    }


def comparison(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    deltas = {}
    for dataset, group in frame.assign(
        _baseline=baseline,
        _candidate=candidate,
    ).groupby("dataset", sort=True):
        delta = (
            roc_auc_score(group["label"], group["_candidate"])
            - roc_auc_score(group["label"], group["_baseline"])
        )
        deltas[str(dataset)] = float(delta)
    values = np.asarray(list(deltas.values()))
    return {
        "wins": int((values > 1e-12).sum()),
        "ties": int((np.abs(values) <= 1e-12).sum()),
        "losses": int((values < -1e-12).sum()),
        "worst_unit_delta": float(values.min()),
        "best_unit_delta": float(values.max()),
        "per_dataset_delta": deltas,
    }


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["direct_logit"] = logit(output["direct_score"].to_numpy())
    output["reasoning_logit"] = logit(output["post_reasoning_score"].to_numpy())
    output["logit_delta"] = output["reasoning_logit"] - output["direct_logit"]
    output["abs_logit_delta"] = output["logit_delta"].abs()
    output["direct_uncertainty"] = output["direct_logit"].abs()
    output["reasoning_uncertainty"] = output["reasoning_logit"].abs()
    output["generated_label"] = [
        int(matches[-1]) if (matches := PREDICTION_RE.findall(text)) else -1
        for text in output["reasoning_reply"].astype(str)
    ]
    output["hard_disagreement"] = (
        (output["direct_score"] >= 0.5)
        != (output["post_reasoning_score"] >= 0.5)
    ).astype(int)
    return output


def gates() -> list[Gate]:
    choices = [
        Gate("always_direct", lambda frame: np.zeros(len(frame), dtype=bool)),
        Gate("always_reasoning", lambda frame: np.ones(len(frame), dtype=bool)),
        Gate(
            "reasoning_more_confident",
            lambda frame: (
                frame["reasoning_uncertainty"] > frame["direct_uncertainty"]
            ).to_numpy(),
        ),
        Gate(
            "hard_disagreement",
            lambda frame: frame["hard_disagreement"].astype(bool).to_numpy(),
        ),
    ]
    for threshold in (32, 64, 96, 128, 192, 256):
        choices.append(Gate(
            f"reasoning_tokens_le_{threshold}",
            lambda frame, threshold=threshold: (
                frame["reasoning_tokens"] <= threshold
            ).to_numpy(),
        ))
    for threshold in (0.25, 0.5, 1.0, 2.0, 4.0):
        choices.append(Gate(
            f"abs_delta_ge_{threshold:g}",
            lambda frame, threshold=threshold: (
                frame["abs_logit_delta"] >= threshold
            ).to_numpy(),
        ))
        choices.append(Gate(
            f"direct_uncertainty_le_{threshold:g}",
            lambda frame, threshold=threshold: (
                frame["direct_uncertainty"] <= threshold
            ).to_numpy(),
        ))
    return choices


def gate_scores(frame: pd.DataFrame, gate: Gate) -> np.ndarray:
    use_reasoning = gate.use_reasoning(frame)
    return np.where(
        use_reasoning,
        frame["post_reasoning_score"].to_numpy(float),
        frame["direct_score"].to_numpy(float),
    )


def balanced_dataset_label_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["dataset", "label"])["label"].transform("size")
    return 1.0 / counts.to_numpy(float)


STACK_FEATURES = (
    "direct_logit",
    "reasoning_logit",
    "abs_logit_delta",
    "reasoning_tokens",
    "hard_disagreement",
)


def fit_stack(frame: pd.DataFrame, c: float) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c,
            max_iter=2_000,
            solver="liblinear",
            random_state=0,
        )),
    ])
    model.fit(
        frame.loc[:, STACK_FEATURES].to_numpy(float),
        frame["label"].to_numpy(int),
        classifier__sample_weight=balanced_dataset_label_weights(frame),
    )
    return model


def grouped_stack_scores(frame: pd.DataFrame, c: float) -> np.ndarray:
    scores = np.full(len(frame), np.nan)
    for dataset in sorted(frame["dataset"].unique()):
        held = frame["dataset"].eq(dataset).to_numpy()
        model = fit_stack(frame.loc[~held].reset_index(drop=True), c)
        scores[held] = model.predict_proba(
            frame.loc[held, STACK_FEATURES].to_numpy(float)
        )[:, 1]
    return scores


def select_stack_c(frame: pd.DataFrame) -> float:
    return max(
        CANDIDATE_CS,
        key=lambda c: (macro_auroc(frame, grouped_stack_scores(frame, c)), -c),
    )


def nested_stack_scores(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    scores = np.full(len(frame), np.nan)
    selections = {}
    for dataset in sorted(frame["dataset"].unique()):
        held = frame["dataset"].eq(dataset).to_numpy()
        train = frame.loc[~held].reset_index(drop=True)
        c = select_stack_c(train)
        model = fit_stack(train, c)
        scores[held] = model.predict_proba(
            frame.loc[held, STACK_FEATURES].to_numpy(float)
        )[:, 1]
        selections[dataset] = c
    return scores, selections


def nested_blend_scores(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    direct = frame["direct_logit"].to_numpy(float)
    reasoning = frame["reasoning_logit"].to_numpy(float)
    output = np.full(len(frame), np.nan)
    selections = {}
    for dataset in sorted(frame["dataset"].unique()):
        held = frame["dataset"].eq(dataset).to_numpy()
        train = frame.loc[~held].reset_index(drop=True)
        weight = max(
            BLEND_WEIGHTS,
            key=lambda value: (
                macro_auroc(
                    train,
                    sigmoid(
                        value * train["direct_logit"].to_numpy(float)
                        + (1.0 - value)
                        * train["reasoning_logit"].to_numpy(float)
                    ),
                ),
                value,
            ),
        )
        output[held] = sigmoid(
            weight * direct[held] + (1.0 - weight) * reasoning[held]
        )
        selections[dataset] = weight
    return output, selections


def nested_gate_scores(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, str]]:
    output = np.full(len(frame), np.nan)
    selections = {}
    candidates = gates()
    for dataset in sorted(frame["dataset"].unique()):
        held = frame["dataset"].eq(dataset).to_numpy()
        train = frame.loc[~held].reset_index(drop=True)
        selected = max(
            candidates,
            key=lambda gate: (
                macro_auroc(train, gate_scores(train, gate)),
                gate.name == "always_direct",
            ),
        )
        output[held] = gate_scores(
            frame.loc[held].reset_index(drop=True),
            selected,
        )
        selections[dataset] = selected.name
    return output, selections


def pairwise_complementarity(frame: pd.DataFrame) -> dict[str, int]:
    recovered = 0
    harmed = 0
    both_correct = 0
    both_wrong = 0
    for _, group in frame.groupby("dataset", sort=True):
        positive = group[group["label"].eq(1)]
        negative = group[group["label"].eq(0)]
        direct = (
            positive["direct_score"].to_numpy()[:, None]
            > negative["direct_score"].to_numpy()[None, :]
        )
        reasoning = (
            positive["post_reasoning_score"].to_numpy()[:, None]
            > negative["post_reasoning_score"].to_numpy()[None, :]
        )
        recovered += int((~direct & reasoning).sum())
        harmed += int((direct & ~reasoning).sum())
        both_correct += int((direct & reasoning).sum())
        both_wrong += int((~direct & ~reasoning).sum())
    return {
        "reasoning_recovers_direct_pair": recovered,
        "reasoning_harms_direct_pair": harmed,
        "both_rank_correctly": both_correct,
        "both_rank_incorrectly_or_tied": both_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--test-input", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = add_features(pd.read_json(args.input, lines=True))
    nested_blend, blend_selections = nested_blend_scores(frame)
    nested_gate, gate_selections = nested_gate_scores(frame)
    nested_stack, stack_selections = nested_stack_scores(frame)
    conditions = {
        "direct": frame["direct_score"].to_numpy(float),
        "post_reasoning": frame["post_reasoning_score"].to_numpy(float),
        "equal_log_odds_blend": sigmoid(
            0.5 * frame["direct_logit"].to_numpy(float)
            + 0.5 * frame["reasoning_logit"].to_numpy(float)
        ),
        "nested_weight_blend": nested_blend,
        "nested_hard_router": nested_gate,
        "nested_soft_router": nested_stack,
    }
    direct = conditions["direct"]
    selected_blend = sigmoid(
        0.6 * frame["direct_logit"].to_numpy(float)
        + 0.4 * frame["reasoning_logit"].to_numpy(float)
    )
    report = {
        "protocol": {
            "evaluation": "nested leave-one-dataset-unit-out",
            "test_used": False,
            "blend_weights": list(BLEND_WEIGHTS),
            "hard_router_candidates": [gate.name for gate in gates()],
            "soft_router_features": list(STACK_FEATURES),
            "soft_router_c": list(CANDIDATE_CS),
        },
        "conditions": {
            name: summarize(frame, scores)
            for name, scores in conditions.items()
        },
        "selected_blend": {
            "direct_weight": 0.6,
            "reasoning_weight": 0.4,
            "comparison_to_direct": comparison(
                frame,
                direct,
                selected_blend,
            ),
        },
        "validation_weight_sweep": {
            str(weight): summarize(
                frame,
                sigmoid(
                    weight * frame["direct_logit"].to_numpy(float)
                    + (1.0 - weight)
                    * frame["reasoning_logit"].to_numpy(float)
                ),
            )
            for weight in BLEND_WEIGHTS
        },
        "pairwise_complementarity": pairwise_complementarity(frame),
        "outer_fold_selections": {
            "blend_direct_weight": blend_selections,
            "hard_router": gate_selections,
            "soft_router_c": stack_selections,
        },
    }
    if args.test_input is not None:
        test = add_features(pd.read_json(args.test_input, lines=True))
        test_direct = test["direct_score"].to_numpy(float)
        test_reasoning = test["post_reasoning_score"].to_numpy(float)
        test_blend = sigmoid(
            0.6 * test["direct_logit"].to_numpy(float)
            + 0.4 * test["reasoning_logit"].to_numpy(float)
        )
        report["frozen_test"] = {
            "direct": summarize(test, test_direct),
            "post_reasoning": summarize(test, test_reasoning),
            "selected_blend": summarize(test, test_blend),
            "selected_blend_comparison_to_direct": comparison(
                test,
                test_direct,
                test_blend,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
