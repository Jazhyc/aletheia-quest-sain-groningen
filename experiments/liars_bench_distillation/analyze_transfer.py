#!/usr/bin/env python3
"""Compare competition and external transfer results against frozen criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def paired_changes(baseline_path: Path, candidate_path: Path) -> dict[str, int]:
    """Count prediction changes, fixes, and breaks on aligned generations."""
    baseline = {
        (row["dataset"], str(row["index"])): row
        for row in map(json.loads, baseline_path.read_text().splitlines())
    }
    candidate = {
        (row["dataset"], str(row["index"])): row
        for row in map(json.loads, candidate_path.read_text().splitlines())
    }
    if baseline.keys() != candidate.keys():
        missing = len(baseline.keys() - candidate.keys())
        extra = len(candidate.keys() - baseline.keys())
        raise ValueError(f"generation keys differ: missing={missing} extra={extra}")
    changes = fixes = breaks = 0
    for key, base in baseline.items():
        other = candidate[key]
        base_prediction = 0 if base.get("prediction") is None else int(base["prediction"])
        other_prediction = (
            0 if other.get("prediction") is None else int(other["prediction"])
        )
        if base_prediction == other_prediction:
            continue
        changes += 1
        label = int(base["label"])
        fixes += int(other_prediction == label)
        breaks += int(base_prediction == label)
    return {"changes": changes, "fixes": fixes, "breaks": breaks}


def metric_delta(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compute top-level and grouped balanced-accuracy deltas."""
    result: dict[str, Any] = {
        "balanced_accuracy": candidate["metrics"]["all"]["balanced_accuracy"],
        "balanced_accuracy_delta": (
            candidate["metrics"]["all"]["balanced_accuracy"]
            - baseline["metrics"]["all"]["balanced_accuracy"]
        ),
        "instructed_delta": (
            candidate["metrics"]["instructed"]["balanced_accuracy"]
            - baseline["metrics"]["instructed"]["balanced_accuracy"]
        ),
        "varied_delta": (
            candidate["metrics"]["varied"]["balanced_accuracy"]
            - baseline["metrics"]["varied"]["balanced_accuracy"]
        ),
        "parse_errors": candidate["parse_errors"],
    }
    return result


def external_delta(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compute external macro and category deltas."""
    categories = sorted(baseline["per_category"])
    return {
        "macro_category_balanced_accuracy": candidate[
            "macro_category_balanced_accuracy"
        ],
        "macro_delta": (
            candidate["macro_category_balanced_accuracy"]
            - baseline["macro_category_balanced_accuracy"]
        ),
        "category_deltas": {
            category: (
                candidate["per_category"][category]["balanced_accuracy"]
                - baseline["per_category"][category]["balanced_accuracy"]
            )
            for category in categories
        },
        "source_model_deltas": {
            model: (
                candidate["per_source_model"][model]["balanced_accuracy"]
                - baseline["per_source_model"][model]["balanced_accuracy"]
            )
            for model in sorted(baseline["per_source_model"])
            if model in candidate["per_source_model"]
        },
    }


def accepts_action_route(
    routed_delta: dict[str, Any],
    router_summary: dict[str, Any],
    *,
    target_category: str = "insider-trading",
    minimum_target_gain: float = 0.10,
) -> bool:
    """Apply the frozen target-gain and zero-spillover action-route gate."""
    non_target_rows = sum(
        count
        for category, count in router_summary["per_category"].items()
        if category != target_category
    )
    return bool(
        non_target_rows == 0
        and routed_delta["category_deltas"][target_category] >= minimum_target_gain
    )


def analyze(
    competition_paths: dict[str, Path],
    external: dict[str, Any],
    *,
    baseline_name: str,
    max_competition_loss: float,
    min_external_gain: float,
) -> dict[str, Any]:
    """Build a report without choosing thresholds after seeing outcomes."""
    competition = {
        name: json.loads(path.read_text()) for name, path in competition_paths.items()
    }
    baseline_competition = competition[baseline_name]
    baseline_external = external["conditions"][baseline_name]
    baseline_routed = external["routed_conditions"][baseline_name]
    baseline_action_routed = external["action_routed_conditions"][baseline_name]
    baseline_content_routed = external["content_routed_conditions"][baseline_name]
    report: dict[str, Any] = {
        "criteria": {
            "max_competition_loss": max_competition_loss,
            "min_external_gain": min_external_gain,
        },
        "baseline": baseline_name,
        "conditions": {},
    }
    baseline_generations = competition_paths[baseline_name].parent / "generations.jsonl"
    for name, result in competition.items():
        comp = metric_delta(result, baseline_competition)
        ext = external_delta(external["conditions"][name], baseline_external)
        routed = external_delta(
            external["routed_conditions"][name], baseline_routed
        )
        action_routed = external_delta(
            external["action_routed_conditions"][name], baseline_action_routed
        )
        content_routed = external_delta(
            external["content_routed_conditions"][name], baseline_content_routed
        )
        generation_path = competition_paths[name].parent / "generations.jsonl"
        paired = (
            {"changes": 0, "fixes": 0, "breaks": 0}
            if name == baseline_name
            else paired_changes(baseline_generations, generation_path)
        )
        report["conditions"][name] = {
            "competition": comp,
            "external": ext,
            "external_routed": routed,
            "external_action_routed": action_routed,
            "external_content_routed": content_routed,
            "paired_competition": paired,
            "accept_general": (
                comp["balanced_accuracy_delta"] >= -max_competition_loss
                and ext["macro_delta"] >= min_external_gain
            ),
            "accept_routed": (
                comp["balanced_accuracy_delta"] >= -max_competition_loss
                and routed["macro_delta"] >= min_external_gain
            ),
            "accept_action_route": (
                accepts_action_route(action_routed, external["action_router"])
            ),
            "accept_content_route": (
                accepts_action_route(content_routed, external["action_router"])
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", action="append", type=parse_named_path, required=True)
    parser.add_argument("--external-result", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--max-competition-loss", type=float, default=0.0025)
    parser.add_argument("--min-external-gain", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = dict(args.competition)
    if len(paths) != len(args.competition):
        raise SystemExit("duplicate competition condition name")
    report = analyze(
        paths,
        json.loads(args.external_result.read_text()),
        baseline_name=args.baseline,
        max_competition_loss=args.max_competition_loss,
        min_external_gain=args.min_external_gain,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
