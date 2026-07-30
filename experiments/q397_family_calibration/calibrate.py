#!/usr/bin/env python3
"""Select family-level Q397 log-odds blends on train and freeze validation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.q397_family_calibration.cache_logits import family_from_text
from experiments.q397_readout_adaptation.run import READOUTS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = (
    ROOT / "results/blackbox/q397_family_calibration_train_cache_v1/scores.jsonl"
)
DEFAULT_VALIDATION = (
    ROOT / "results/blackbox/q397_auxiliary_readout_validation_v1/scores.jsonl"
)
DEFAULT_INTENT = (
    ROOT
    / "results/blackbox/specialist_necessity_ndif_v1/"
    "intent_validation_current_rank1.csv"
)
DEFAULT_OUTPUT = ROOT / "results/blackbox/q397_family_calibration_v1"
DIGIT_WEIGHT_GRID = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5)
MIN_TRAIN_GAIN = 0.001
MIN_NONWORSE_FRACTION = 0.75
MAX_TRAIN_UNIT_LOSS = 0.01
MIN_SCENARIO_DELTA = -0.002
VALIDATION_MACRO_GAIN = 0.001
VALIDATION_FAMILY_FLOOR = -0.002
VALIDATION_SCENARIO_FLOOR = -0.001


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines score cache and infer missing family labels."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for row in rows:
        row.setdefault("family", family_from_text(str(row["dataset"])))
    return rows


def logit(probability: float) -> float:
    """Convert a bounded probability into log odds."""
    bounded = min(max(float(probability), 1.0e-12), 1.0 - 1.0e-12)
    return math.log(bounded) - math.log1p(-bounded)


def sigmoid(value: float) -> float:
    """Return a numerically stable probability."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


def score_margin(row: dict[str, Any], readout: str) -> float:
    """Recover one readout's exact positive-minus-negative logit margin."""
    token_pairs = {
        "digits": (15, 16),
        "no_yes": (2665, 9175),
        "false_true": (3913, 2434),
        "a_b": (32, 33),
        "honest_deceptive": (10429, 77507),
        "truthful_misleading": (86775, 35978),
    }
    negative, positive = token_pairs[readout]
    values = row["requested_logprobs"]
    return float(values[str(positive)]) - float(values[str(negative)])


def blended_margin(
    row: dict[str, Any],
    *,
    auxiliary: str | None,
    digit_weight: float,
) -> float:
    """Blend digit and auxiliary margins without another model call."""
    digit = score_margin(row, "digits")
    if auxiliary is None or digit_weight == 1.0:
        return digit
    return (
        digit_weight * digit
        + (1.0 - digit_weight) * score_margin(row, auxiliary)
    )


def per_dataset_auroc(
    rows: Iterable[dict[str, Any]], scores: Iterable[float]
) -> dict[str, float]:
    """Compute AUROC independently for every competition dataset unit."""
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


def macro_auroc(rows: list[dict[str, Any]], scores: list[float]) -> float:
    """Return the competition mean per-dataset AUROC."""
    values = per_dataset_auroc(rows, scores)
    return float(np.mean(list(values.values())))


def scenario_macro(
    rows: list[dict[str, Any]], scores: list[float]
) -> dict[str, float]:
    """Report macro AUROC by public scenario."""
    result = {}
    for scenario in ("instructed-deception", "varied-deception"):
        offsets = [
            offset
            for offset, row in enumerate(rows)
            if scenario in str(row["dataset"])
        ]
        if offsets:
            result[scenario] = macro_auroc(
                [rows[offset] for offset in offsets],
                [scores[offset] for offset in offsets],
            )
    return result


def candidate_scores(
    rows: list[dict[str, Any]], auxiliary: str | None, digit_weight: float
) -> list[float]:
    """Score rows for one fixed calibration candidate."""
    return [
        sigmoid(
            blended_margin(
                row, auxiliary=auxiliary, digit_weight=digit_weight
            )
        )
        for row in rows
    ]


def select_family_parameter(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a conservative family parameter from training labels only."""
    baseline_scores = candidate_scores(rows, None, 1.0)
    baseline_units = per_dataset_auroc(rows, baseline_scores)
    baseline_macro = float(np.mean(list(baseline_units.values())))
    baseline_scenarios = scenario_macro(rows, baseline_scores)
    unit_count = len(baseline_units)
    candidates: list[dict[str, Any]] = []
    auxiliary_names = [readout.name for readout in READOUTS if readout.name != "digits"]
    for auxiliary in auxiliary_names:
        for digit_weight in DIGIT_WEIGHT_GRID:
            scores = candidate_scores(rows, auxiliary, digit_weight)
            unit_scores = per_dataset_auroc(rows, scores)
            deltas = [
                unit_scores[dataset] - baseline_units[dataset]
                for dataset in sorted(baseline_units)
            ]
            scenarios = scenario_macro(rows, scores)
            scenario_deltas = {
                name: value - baseline_scenarios[name]
                for name, value in scenarios.items()
            }
            macro = float(np.mean(list(unit_scores.values())))
            nonworse = sum(delta >= -1.0e-12 for delta in deltas)
            eligible = bool(
                unit_count >= 2
                and macro - baseline_macro >= MIN_TRAIN_GAIN
                and nonworse / unit_count >= MIN_NONWORSE_FRACTION
                and min(deltas) >= -MAX_TRAIN_UNIT_LOSS
                and min(scenario_deltas.values()) >= MIN_SCENARIO_DELTA
            )
            candidates.append(
                {
                    "auxiliary": auxiliary,
                    "digit_weight": digit_weight,
                    "macro_auroc": macro,
                    "delta": macro - baseline_macro,
                    "nonworse_units": nonworse,
                    "unit_count": unit_count,
                    "worst_unit_delta": min(deltas),
                    "scenario_deltas": scenario_deltas,
                    "eligible": eligible,
                }
            )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return (
            {
                "auxiliary": None,
                "digit_weight": 1.0,
                "macro_auroc": baseline_macro,
                "delta": 0.0,
                "reason": "no candidate passed the train robustness gates",
            },
            candidates,
        )
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["macro_auroc"],
            candidate["digit_weight"],
            candidate["auxiliary"],
        ),
    )
    return selected, candidates


def select_parameters(
    train_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Select exactly one family-wide parameter for each represented family."""
    parameters = {}
    searches = {}
    for family in sorted({str(row["family"]) for row in train_rows}):
        rows = [row for row in train_rows if row["family"] == family]
        parameters[family], searches[family] = select_family_parameter(rows)
    return parameters, searches


def empirical_parameters(
    robust_parameters: dict[str, dict[str, Any]],
    searches: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Choose pure train-ERM parameters, retaining one-unit families unchanged."""
    parameters = {}
    for family, candidates in searches.items():
        if not candidates or int(candidates[0]["unit_count"]) < 2:
            parameters[family] = robust_parameters[family]
            continue
        parameters[family] = max(
            candidates,
            key=lambda candidate: (
                candidate["macro_auroc"],
                candidate["digit_weight"],
                candidate["auxiliary"],
            ),
        )
    return parameters


def apply_parameters(
    rows: list[dict[str, Any]],
    parameters: dict[str, dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Return uncalibrated and family-calibrated Q397 margins as scores."""
    baseline = candidate_scores(rows, None, 1.0)
    calibrated = []
    for row in rows:
        parameter = parameters[str(row["family"])]
        calibrated.append(
            sigmoid(
                blended_margin(
                    row,
                    auxiliary=parameter["auxiliary"],
                    digit_weight=float(parameter["digit_weight"]),
                )
            )
        )
    return baseline, calibrated


def load_intent_scores(path: Path) -> dict[tuple[str, str], float]:
    """Load the frozen rank-1 intent member used by current Phoenix."""
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            (str(row["dataset"]), str(row["index"])): float(row["score"])
            for row in csv.DictReader(handle)
        }


def overlay_intent(
    rows: list[dict[str, Any]],
    scores: list[float],
    intent_scores: dict[tuple[str, str], float],
) -> list[float]:
    """Apply Phoenix's fixed mean-log-odds intent blend where available."""
    result = []
    for row, score in zip(rows, scores, strict=True):
        key = (str(row["dataset"]), str(row["index"]))
        if key not in intent_scores:
            result.append(score)
            continue
        result.append(sigmoid(0.5 * (logit(score) + logit(intent_scores[key]))))
    return result


def summarize(
    rows: list[dict[str, Any]],
    baseline: list[float],
    candidate: list[float],
) -> dict[str, Any]:
    """Summarize a frozen candidate against its matched baseline."""
    baseline_macro = macro_auroc(rows, baseline)
    candidate_macro = macro_auroc(rows, candidate)
    baseline_scenarios = scenario_macro(rows, baseline)
    candidate_scenarios = scenario_macro(rows, candidate)
    families = {}
    for family in sorted({str(row["family"]) for row in rows}):
        offsets = [
            offset for offset, row in enumerate(rows) if row["family"] == family
        ]
        family_rows = [rows[offset] for offset in offsets]
        base_value = macro_auroc(
            family_rows, [baseline[offset] for offset in offsets]
        )
        candidate_value = macro_auroc(
            family_rows, [candidate[offset] for offset in offsets]
        )
        families[family] = {
            "baseline": base_value,
            "candidate": candidate_value,
            "delta": candidate_value - base_value,
        }
    scenario_deltas = {
        name: candidate_scenarios[name] - baseline_scenarios[name]
        for name in baseline_scenarios
    }
    passes = bool(
        candidate_macro - baseline_macro >= VALIDATION_MACRO_GAIN
        and min(item["delta"] for item in families.values())
        >= VALIDATION_FAMILY_FLOOR
        and min(scenario_deltas.values()) >= VALIDATION_SCENARIO_FLOOR
    )
    return {
        "baseline_macro_auroc": baseline_macro,
        "candidate_macro_auroc": candidate_macro,
        "delta": candidate_macro - baseline_macro,
        "baseline_scenarios": baseline_scenarios,
        "candidate_scenarios": candidate_scenarios,
        "scenario_deltas": scenario_deltas,
        "families": families,
        "unique_scores": {
            "baseline": int(np.unique(baseline).size),
            "candidate": int(np.unique(candidate).size),
        },
        "passes_validation_gate": passes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Select on train, freeze, and evaluate once on validation."""
    train_rows = load_jsonl(args.train_scores)
    validation_rows = load_jsonl(args.validation_scores)
    parameters, searches = select_parameters(train_rows)
    train_argmax_parameters = empirical_parameters(parameters, searches)
    baseline, calibrated = apply_parameters(validation_rows, parameters)
    _, train_argmax_calibrated = apply_parameters(
        validation_rows, train_argmax_parameters
    )
    intent = load_intent_scores(args.intent_scores)
    phoenix_baseline = overlay_intent(validation_rows, baseline, intent)
    phoenix_candidate = overlay_intent(validation_rows, calibrated, intent)
    phoenix_train_argmax = overlay_intent(
        validation_rows, train_argmax_calibrated, intent
    )
    report = {
        "method": "q397_family_calibration_v1",
        "principle": (
            "family-level same-position log-odds blend; no organism-specific parameter"
        ),
        "selection_split": "train",
        "evaluation_split": "validation",
        "grids": {
            "digit_weights": list(DIGIT_WEIGHT_GRID),
            "auxiliary_readouts": [
                readout.name for readout in READOUTS if readout.name != "digits"
            ],
        },
        "train_gates": {
            "minimum_macro_gain": MIN_TRAIN_GAIN,
            "minimum_nonworse_fraction": MIN_NONWORSE_FRACTION,
            "maximum_unit_loss": MAX_TRAIN_UNIT_LOSS,
            "minimum_scenario_delta": MIN_SCENARIO_DELTA,
            "minimum_units": 2,
        },
        "validation_gates": {
            "minimum_macro_gain": VALIDATION_MACRO_GAIN,
            "minimum_family_delta": VALIDATION_FAMILY_FLOOR,
            "minimum_scenario_delta": VALIDATION_SCENARIO_FLOOR,
        },
        "selected_parameters": parameters,
        "exploratory_train_argmax_parameters": train_argmax_parameters,
        "train_searches": searches,
        "validation_q397_only": summarize(
            validation_rows, baseline, calibrated
        ),
        "validation_current_phoenix": summarize(
            validation_rows, phoenix_baseline, phoenix_candidate
        ),
        "validation_train_argmax_q397_only": summarize(
            validation_rows, baseline, train_argmax_calibrated
        ),
        "validation_train_argmax_current_phoenix": summarize(
            validation_rows, phoenix_baseline, phoenix_train_argmax
        ),
        "intent_rows": len(intent),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output_dir / "validation_scores.jsonl").open("w") as handle:
        for row, base, candidate, train_argmax in zip(
            validation_rows,
            phoenix_baseline,
            phoenix_candidate,
            phoenix_train_argmax,
            strict=True,
        ):
            handle.write(
                json.dumps(
                    {
                        "dataset": row["dataset"],
                        "family": row["family"],
                        "index": row["index"],
                        "label": row["label"],
                        "baseline": base,
                        "candidate": candidate,
                        "train_argmax_candidate": train_argmax,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scores", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-scores", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--intent-scores", type=Path, default=DEFAULT_INTENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
