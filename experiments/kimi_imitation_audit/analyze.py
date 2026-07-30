#!/usr/bin/env python3
"""Analyze whether the Kimi student fits its training-set teacher ordering."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHOD = "kimi_k3_student_train_imitation_audit_v1"
DEFAULT_SCORES = ROOT / "results/blackbox" / DEFAULT_METHOD / "scores.jsonl"


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load non-empty, uniquely keyed score rows."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("score artifact is empty")
    keys = [(str(row["dataset"]), str(row["index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate dataset/index keys")
    return rows


def safe_correlation(function: Any, left: np.ndarray, right: np.ndarray) -> float:
    """Compute a correlation, failing explicitly on a degenerate result."""
    result = function(left, right)
    value = float(result.statistic)
    if not np.isfinite(value):
        raise ValueError("correlation is undefined")
    return value


def affine_fit(
    teacher_margins: np.ndarray,
    student_margins: np.ndarray,
) -> dict[str, float]:
    """Fit the best affine student-margin approximation to teacher margins."""
    design = np.column_stack(
        [teacher_margins, np.ones_like(teacher_margins)]
    )
    slope, intercept = np.linalg.lstsq(
        design,
        student_margins,
        rcond=None,
    )[0]
    predictions = slope * teacher_margins + intercept
    residuals = student_margins - predictions
    total = student_margins - student_margins.mean()
    denominator = float(np.dot(total, total))
    r_squared = (
        1.0 - float(np.dot(residuals, residuals)) / denominator
        if denominator > 0.0
        else 0.0
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "residual_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
    }


def pair_order_agreement(
    teacher: np.ndarray,
    student: np.ndarray,
) -> dict[str, float | int]:
    """Compare all unordered within-unit pairs, accounting for ties."""
    upper = np.triu_indices(len(teacher), k=1)
    teacher_differences = teacher[:, None] - teacher[None, :]
    student_differences = student[:, None] - student[None, :]
    teacher_sign = np.sign(teacher_differences[upper])
    student_sign = np.sign(student_differences[upper])
    informative = teacher_sign != 0
    exact = student_sign[informative] == teacher_sign[informative]
    return {
        "pairs": int(len(teacher_sign)),
        "teacher_informative_pairs": int(informative.sum()),
        "direction_agreement": float(exact.mean()),
        "student_tie_rate_on_teacher_informative": float(
            np.mean(student_sign[informative] == 0)
        ),
    }


def cross_label_pair_agreement(
    labels: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
) -> dict[str, float | int]:
    """Compare teacher/student ordering for every positive-negative pair."""
    positive = labels == 1
    negative = labels == 0
    teacher_differences = teacher[positive, None] - teacher[None, negative]
    student_differences = student[positive, None] - student[None, negative]
    teacher_sign = np.sign(teacher_differences.ravel())
    student_sign = np.sign(student_differences.ravel())
    informative = teacher_sign != 0
    return {
        "pairs": int(teacher_sign.size),
        "teacher_informative_pairs": int(informative.sum()),
        "direction_agreement": float(
            np.mean(student_sign[informative] == teacher_sign[informative])
        ),
        "teacher_correct_rate": float(
            np.mean(teacher_sign > 0) + 0.5 * np.mean(teacher_sign == 0)
        ),
        "student_correct_rate": float(
            np.mean(student_sign > 0) + 0.5 * np.mean(student_sign == 0)
        ),
    }


def summarize_unit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize imitation and label ranking for one dataset unit."""
    labels = np.asarray([int(row["label"]) for row in rows])
    teacher_scores = np.asarray(
        [float(row["teacher_probability"]) for row in rows]
    )
    student_scores = np.asarray(
        [float(row["student_probability"]) for row in rows]
    )
    base_scores = np.asarray([float(row["base_probability"]) for row in rows])
    teacher_margins = np.asarray(
        [float(row["teacher_margin"]) for row in rows]
    )
    student_margins = np.asarray(
        [float(row["student_margin"]) for row in rows]
    )
    return {
        "rows": len(rows),
        "teacher_auroc": float(roc_auc_score(labels, teacher_scores)),
        "student_auroc": float(roc_auc_score(labels, student_scores)),
        "base_auroc": float(roc_auc_score(labels, base_scores)),
        "probability_mae": float(np.mean(np.abs(student_scores - teacher_scores))),
        "margin_mae": float(np.mean(np.abs(student_margins - teacher_margins))),
        "margin_rmse": float(
            np.sqrt(np.mean(np.square(student_margins - teacher_margins)))
        ),
        "margin_pearson": safe_correlation(
            pearsonr, teacher_margins, student_margins
        ),
        "margin_spearman": safe_correlation(
            spearmanr, teacher_margins, student_margins
        ),
        "margin_kendall_tau": safe_correlation(
            kendalltau, teacher_margins, student_margins
        ),
        "sign_agreement": float(
            np.mean(np.sign(teacher_margins) == np.sign(student_margins))
        ),
        "all_pair_order": pair_order_agreement(
            teacher_margins, student_margins
        ),
        "cross_label_pair_order": cross_label_pair_agreement(
            labels, teacher_margins, student_margins
        ),
    }


def margin_bin_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Measure residual error by teacher confidence magnitude."""
    bins = (
        ("abs_margin_lt_1", 0.0, 1.0),
        ("abs_margin_1_to_3", 1.0, 3.0),
        ("abs_margin_3_to_5", 3.0, 5.0),
        ("abs_margin_ge_5", 5.0, float("inf")),
    )
    result = {}
    for name, lower, upper in bins:
        selected = [
            row
            for row in rows
            if lower <= abs(float(row["teacher_margin"])) < upper
        ]
        if not selected:
            continue
        teacher_margin = np.asarray(
            [float(row["teacher_margin"]) for row in selected]
        )
        student_margin = np.asarray(
            [float(row["student_margin"]) for row in selected]
        )
        teacher_probability = np.asarray(
            [float(row["teacher_probability"]) for row in selected]
        )
        student_probability = np.asarray(
            [float(row["student_probability"]) for row in selected]
        )
        result[name] = {
            "rows": len(selected),
            "probability_mae": float(
                np.mean(np.abs(student_probability - teacher_probability))
            ),
            "margin_mae": float(np.mean(np.abs(student_margin - teacher_margin))),
            "sign_agreement": float(
                np.mean(np.sign(student_margin) == np.sign(teacher_margin))
            ),
        }
    return result


def label_teacher_side_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Describe student behavior where teacher sign agrees/conflicts with labels."""
    result = {}
    for label in (0, 1):
        for teacher_correct in (True, False):
            selected = []
            for row in rows:
                if int(row["label"]) != label:
                    continue
                margin = float(row["teacher_margin"])
                correct = margin < 0 if label == 0 else margin > 0
                if correct == teacher_correct:
                    selected.append(row)
            name = f"label_{label}_teacher_{'correct' if teacher_correct else 'wrong'}"
            if not selected:
                continue
            result[name] = {
                "rows": len(selected),
                "teacher_mean_margin": float(
                    np.mean([float(row["teacher_margin"]) for row in selected])
                ),
                "student_mean_margin": float(
                    np.mean([float(row["student_margin"]) for row in selected])
                ),
                "teacher_student_sign_agreement": float(
                    np.mean([
                        np.sign(float(row["teacher_margin"]))
                        == np.sign(float(row["student_margin"]))
                        for row in selected
                    ])
                ),
                "student_label_side_rate": float(
                    np.mean([
                        float(row["student_margin"]) < 0
                        if label == 0
                        else float(row["student_margin"]) > 0
                        for row in selected
                    ])
                ),
            }
    return result


def render_markdown(report: dict[str, Any]) -> str:
    """Render the main audit findings."""
    aggregate = report["aggregate"]
    lines = [
        "| metric | base | student | teacher |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| train macro AUROC | {aggregate['base_macro_auroc']:.6f} | "
            f"{aggregate['student_macro_auroc']:.6f} | "
            f"{aggregate['teacher_macro_auroc']:.6f} |"
        ),
        (
            f"| delta over base | — | {aggregate['student_delta_over_base']:+.6f} | "
            f"{aggregate['teacher_delta_over_base']:+.6f} |"
        ),
        "",
        (
            "- Fraction of Kimi's AUROC gain over base retained by the student: "
            f"`{aggregate['teacher_gain_retained']:.2%}`"
        ),
        f"- Student–teacher macro gap: `{aggregate['teacher_student_macro_gap']:+.6f}`",
        f"- Mean within-unit Spearman: `{aggregate['mean_unit_spearman']:.6f}`",
        (
            "- Mean all-pair direction agreement: "
            f"`{aggregate['mean_unit_all_pair_agreement']:.6f}`"
        ),
        (
            "- Mean cross-label teacher/student direction agreement: "
            f"`{aggregate['mean_unit_cross_label_pair_agreement']:.6f}`"
        ),
        (
            "- Global affine margin fit: "
            f"slope `{report['global_affine_fit']['slope']:.4f}`, "
            f"intercept `{report['global_affine_fit']['intercept']:.4f}`, "
            f"R² `{report['global_affine_fit']['r_squared']:.6f}`"
        ),
        f"- Audit conclusion: **{report['conclusion']['status']}**",
        "",
        report["conclusion"]["explanation"],
        "",
    ]
    return "\n".join(lines)


def analyze(path: Path) -> dict[str, Any]:
    """Compute unit-weighted imitation and optimization diagnostics."""
    rows = load_rows(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    units = {
        dataset: summarize_unit(unit_rows)
        for dataset, unit_rows in sorted(grouped.items())
    }

    teacher_macro = float(np.mean([unit["teacher_auroc"] for unit in units.values()]))
    student_macro = float(np.mean([unit["student_auroc"] for unit in units.values()]))
    base_macro = float(np.mean([unit["base_auroc"] for unit in units.values()]))
    teacher_delta = teacher_macro - base_macro
    student_delta = student_macro - base_macro
    mean_spearman = float(
        np.mean([unit["margin_spearman"] for unit in units.values()])
    )
    mean_all_pair = float(
        np.mean([
            unit["all_pair_order"]["direction_agreement"]
            for unit in units.values()
        ])
    )
    mean_cross_label_pair = float(
        np.mean([
            unit["cross_label_pair_order"]["direction_agreement"]
            for unit in units.values()
        ])
    )
    gap = teacher_macro - student_macro
    if gap <= 0.005 and mean_spearman >= 0.98 and mean_all_pair >= 0.95:
        status = "optimization is not the primary bottleneck"
        explanation = (
            "The student closely reproduces Kimi's in-domain ordering; changing "
            "pointwise loss geometry is unlikely to close the held-out gap."
        )
    elif gap >= 0.01 and (mean_spearman < 0.97 or mean_all_pair < 0.90):
        status = "optimization or adapter capacity is a material bottleneck"
        explanation = (
            "The student leaves substantial Kimi ordering information unfitted "
            "on the training distribution, so objective or capacity changes are "
            "supported before acquiring more supervision."
        )
    else:
        status = "mixed optimization and generalization bottlenecks"
        explanation = (
            "The student transfers much, but not all, of Kimi's training-set "
            "ordering. A single matched objective ablation is justified, while "
            "held-out distribution shift remains material."
        )

    teacher_margins = np.asarray(
        [float(row["teacher_margin"]) for row in rows]
    )
    student_margins = np.asarray(
        [float(row["student_margin"]) for row in rows]
    )
    report = {
        "method": DEFAULT_METHOD,
        "rows": len(rows),
        "dataset_units": len(units),
        "aggregate": {
            "base_macro_auroc": base_macro,
            "student_macro_auroc": student_macro,
            "teacher_macro_auroc": teacher_macro,
            "student_delta_over_base": student_delta,
            "teacher_delta_over_base": teacher_delta,
            "teacher_gain_retained": (
                student_delta / teacher_delta
                if abs(teacher_delta) > 1.0e-12
                else None
            ),
            "teacher_student_macro_gap": gap,
            "mean_unit_pearson": float(
                np.mean([unit["margin_pearson"] for unit in units.values()])
            ),
            "mean_unit_spearman": mean_spearman,
            "mean_unit_kendall_tau": float(
                np.mean([unit["margin_kendall_tau"] for unit in units.values()])
            ),
            "mean_unit_all_pair_agreement": mean_all_pair,
            "mean_unit_cross_label_pair_agreement": mean_cross_label_pair,
            "mean_unit_probability_mae": float(
                np.mean([unit["probability_mae"] for unit in units.values()])
            ),
            "mean_unit_margin_mae": float(
                np.mean([unit["margin_mae"] for unit in units.values()])
            ),
        },
        "global_affine_fit": affine_fit(teacher_margins, student_margins),
        "margin_bins": margin_bin_summary(rows),
        "label_teacher_side": label_teacher_side_summary(rows),
        "per_dataset": units,
        "conclusion": {
            "status": status,
            "explanation": explanation,
            "frozen_thresholds": {
                "not_primary_max_macro_gap": 0.005,
                "not_primary_min_mean_unit_spearman": 0.98,
                "not_primary_min_all_pair_agreement": 0.95,
                "material_min_macro_gap": 0.01,
                "material_max_mean_unit_spearman": 0.97,
                "material_max_all_pair_agreement": 0.90,
            },
        },
    }
    path.with_name("summary.md").write_text(render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.scores)
    output = args.output or args.scores.with_name("summary.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.scores.with_name("summary.md").read_text(), end="")


if __name__ == "__main__":
    main()
