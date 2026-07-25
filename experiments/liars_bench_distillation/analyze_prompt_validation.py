#!/usr/bin/env python3
"""Apply frozen competition-preservation gates to a confirmed prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_distillation.analyze_transfer import (
    metric_delta,
    paired_changes,
)


def analyze_validation(
    control: dict[str, Any],
    candidate: dict[str, Any],
    confirmation: dict[str, Any],
    *,
    candidate_name: str,
    maximum_balanced_accuracy_loss: float,
    maximum_scenario_loss: float,
    maximum_parse_error_increase: int,
    control_generations: Path | None = None,
    candidate_generations: Path | None = None,
) -> dict[str, Any]:
    """Report whether an externally confirmed prompt preserves validation."""
    competition = metric_delta(candidate, control)
    parse_error_increase = int(candidate["parse_errors"]) - int(
        control["parse_errors"]
    )
    confirmed = confirmation.get("selected") == candidate_name
    accepted = bool(
        confirmed
        and competition["balanced_accuracy_delta"]
        >= -maximum_balanced_accuracy_loss
        and competition["instructed_delta"] >= -maximum_scenario_loss
        and competition["varied_delta"] >= -maximum_scenario_loss
        and parse_error_increase <= maximum_parse_error_increase
    )
    report: dict[str, Any] = {
        "candidate": candidate_name,
        "externally_confirmed": confirmed,
        "criteria": {
            "maximum_balanced_accuracy_loss": maximum_balanced_accuracy_loss,
            "maximum_scenario_loss": maximum_scenario_loss,
            "maximum_parse_error_increase": maximum_parse_error_increase,
        },
        "competition": competition,
        "parse_error_increase": parse_error_increase,
        "accepted": accepted,
    }
    if control_generations is not None and candidate_generations is not None:
        report["paired"] = paired_changes(
            control_generations,
            candidate_generations,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--control-generations", type=Path)
    parser.add_argument("--candidate-generations", type=Path)
    parser.add_argument("--maximum-balanced-accuracy-loss", type=float, default=0.0025)
    parser.add_argument("--maximum-scenario-loss", type=float, default=0.01)
    parser.add_argument("--maximum-parse-error-increase", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_validation(
        json.loads(args.control_result.read_text()),
        json.loads(args.candidate_result.read_text()),
        json.loads(args.confirmation.read_text()),
        candidate_name=args.candidate_name,
        maximum_balanced_accuracy_loss=args.maximum_balanced_accuracy_loss,
        maximum_scenario_loss=args.maximum_scenario_loss,
        maximum_parse_error_increase=args.maximum_parse_error_increase,
        control_generations=args.control_generations,
        candidate_generations=args.candidate_generations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
