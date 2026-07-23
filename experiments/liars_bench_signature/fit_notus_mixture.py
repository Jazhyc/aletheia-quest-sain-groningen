#!/usr/bin/env python3
"""Fit exploratory convex mixtures of Liars' Bench signatures to Notus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/blackbox/liars_bench_frozen_judge_signatures_v1/result.json"
OUTPUT = RESULT.with_name("notus_mixture_fit.json")
CONDITIONS = ("truth", "details", "omission", "phoenix_v21_adapter")
NOTUS = {
    "nemotron": {"recall": (.495, .455, .415, .580), "fpr": (.075, .030, .020, .085)},
    "qwen": {"recall": (.375, .405, .305, .515), "fpr": (.050, .065, .040, .100)},
    "gemma": {"recall": (.480, .450, .490, .620), "fpr": (.060, .045, .020, .130)},
}


def fit_simplex(matrix: np.ndarray, target: np.ndarray, ridge: float = 0.01) -> dict[str, Any]:
    n = matrix.shape[1]
    uniform = np.full(n, 1 / n)
    objective = lambda weights: float(np.sum((matrix @ weights - target) ** 2) + ridge * np.sum((weights - uniform) ** 2))
    fitted = minimize(objective, uniform, bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1})
    single_rmse = np.sqrt(np.mean((matrix - target[:, None]) ** 2, axis=0))
    return {
        "weights": fitted.x.tolist(),
        "prediction": (matrix @ fitted.x).tolist(),
        "target": target.tolist(),
        "rmse": float(np.sqrt(np.mean((matrix @ fitted.x - target) ** 2))),
        "best_single_category_index": int(np.argmin(single_rmse)),
        "best_single_category_rmse": float(np.min(single_rmse)),
        "success": bool(fitted.success),
    }


def main() -> None:
    result = json.loads(RESULT.read_text())
    categories = [
        category for category in result["populations"]
        if set(result["populations"][category]["labels"]) == {"0", "1"}
    ]
    signatures = {
        metric: np.array([
            [result["conditions"][condition]["categories"][category]["metrics"][metric] for category in categories]
            for condition in CONDITIONS
        ])
        for metric in ("recall", "fpr")
    }
    output: dict[str, Any] = {"categories": categories, "conditions": list(CONDITIONS), "ridge": 0.01, "fits": {}}
    for metric, matrix in signatures.items():
        output["fits"][metric] = {
            organism: fit_simplex(matrix, np.asarray(values[metric])) for organism, values in NOTUS.items()
        }
        shared_target = np.concatenate([np.asarray(values[metric]) for values in NOTUS.values()])
        shared_matrix = np.concatenate([matrix] * len(NOTUS), axis=0)
        output["fits"][metric]["shared"] = fit_simplex(shared_matrix, shared_target)
    joint_matrix = np.concatenate([signatures["recall"], signatures["fpr"]], axis=0)
    output["fits"]["joint"] = {
        organism: fit_simplex(
            joint_matrix,
            np.concatenate([np.asarray(values["recall"]), np.asarray(values["fpr"])]),
        )
        for organism, values in NOTUS.items()
    }
    shared_joint_target = np.concatenate([
        np.concatenate([np.asarray(values["recall"]), np.asarray(values["fpr"])])
        for values in NOTUS.values()
    ])
    output["fits"]["joint"]["shared"] = fit_simplex(
        np.concatenate([joint_matrix] * len(NOTUS), axis=0), shared_joint_target
    )
    cross_validation = {}
    for held_index, held_condition in enumerate(CONDITIONS):
        test_rows = [held_index, len(CONDITIONS) + held_index]
        train_rows = [index for index in range(len(CONDITIONS) * 2) if index not in test_rows]
        train_matrix = np.concatenate([joint_matrix[train_rows]] * len(NOTUS), axis=0)
        test_matrix = np.concatenate([joint_matrix[test_rows]] * len(NOTUS), axis=0)
        organism_targets = [
            np.concatenate([np.asarray(values["recall"]), np.asarray(values["fpr"])])
            for values in NOTUS.values()
        ]
        train_target = np.concatenate([target[train_rows] for target in organism_targets])
        test_target = np.concatenate([target[test_rows] for target in organism_targets])
        trained = fit_simplex(train_matrix, train_target)
        weights = np.asarray(trained["weights"])
        prediction = test_matrix @ weights
        single_index = trained["best_single_category_index"]
        cross_validation[held_condition] = {
            "weights": weights.tolist(),
            "test_prediction": prediction.tolist(),
            "test_target": test_target.tolist(),
            "test_rmse": float(np.sqrt(np.mean((prediction - test_target) ** 2))),
            "selected_single_category_index": single_index,
            "selected_single_test_rmse": float(np.sqrt(np.mean((test_matrix[:, single_index] - test_target) ** 2))),
        }
    output["joint_shared_leave_one_condition_out"] = cross_validation
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
