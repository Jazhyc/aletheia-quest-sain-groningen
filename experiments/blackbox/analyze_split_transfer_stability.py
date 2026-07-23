#!/usr/bin/env python3
"""Audit whether a judge improvement transfers across matched dataset units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def dataset_ba(result: dict[str, Any]) -> dict[str, float]:
    """Return one balanced-accuracy value per dataset unit."""
    values = {
        str(row["dataset"]): float(row["metrics"]["balanced_accuracy"])
        for row in result["datasets"]
    }
    if len(values) != len(result["datasets"]):
        raise ValueError("duplicate dataset unit in result")
    return values


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def compare_transfer(
    baseline_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
    baseline_test: dict[str, Any],
    candidate_test: dict[str, Any],
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 20260717,
) -> dict[str, Any]:
    """Compare matched per-unit deltas across validation and test."""
    maps = [
        dataset_ba(result)
        for result in (
            baseline_validation,
            candidate_validation,
            baseline_test,
            candidate_test,
        )
    ]
    keys = set(maps[0])
    if any(set(values) != keys for values in maps[1:]):
        raise ValueError("dataset-unit keys differ across results")
    ordered = sorted(keys)
    validation = np.asarray([maps[1][key] - maps[0][key] for key in ordered])
    test = np.asarray([maps[3][key] - maps[2][key] for key in ordered])

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ordered), size=(bootstrap_samples, len(ordered)))
    validation_means = validation[indices].mean(axis=1)
    test_means = test[indices].mean(axis=1)

    epsilon = 1e-12
    validation_sign = np.sign(np.where(np.abs(validation) <= epsilon, 0.0, validation))
    test_sign = np.sign(np.where(np.abs(test) <= epsilon, 0.0, test))
    exact_sign_agreement = validation_sign == test_sign
    nonopposed = validation * test >= -epsilon
    correlation = None
    if np.std(validation) > epsilon and np.std(test) > epsilon:
        correlation = float(np.corrcoef(validation, test)[0, 1])

    unit_rows = [
        {
            "dataset": key,
            "validation_delta": float(validation[index]),
            "test_delta": float(test[index]),
        }
        for index, key in enumerate(ordered)
    ]
    return {
        "n_dataset_units": len(ordered),
        "validation_macro_delta": float(validation.mean()),
        "test_macro_delta": float(test.mean()),
        "validation_unit_bootstrap_95": percentile_interval(validation_means),
        "test_unit_bootstrap_95": percentile_interval(test_means),
        "unit_delta_pearson": correlation,
        "exact_sign_agreement_fraction": float(exact_sign_agreement.mean()),
        "nonopposed_fraction": float(nonopposed.mean()),
        "wins_both": int(((validation > epsilon) & (test > epsilon)).sum()),
        "losses_both": int(((validation < -epsilon) & (test < -epsilon)).sum()),
        "direction_reversals": int((validation * test < -epsilon).sum()),
        "validation_zero_units": int((np.abs(validation) <= epsilon).sum()),
        "test_zero_units": int((np.abs(test) <= epsilon).sum()),
        "units": unit_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-validation", type=Path, required=True)
    parser.add_argument("--candidate-validation", type=Path, required=True)
    parser.add_argument("--baseline-test", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        json.loads(path.read_text())
        for path in (
            args.baseline_validation,
            args.candidate_validation,
            args.baseline_test,
            args.candidate_test,
        )
    ]
    result = compare_transfer(*inputs)
    result["inputs"] = {
        "baseline_validation": args.baseline_validation.as_posix(),
        "candidate_validation": args.candidate_validation.as_posix(),
        "baseline_test": args.baseline_test.as_posix(),
        "candidate_test": args.candidate_test.as_posix(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
