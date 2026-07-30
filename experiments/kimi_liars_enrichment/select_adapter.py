#!/usr/bin/env python3
"""Apply the frozen Phoenix 8.1 competition/Liars promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAX_MACRO_REGRESSION = 0.001
MAX_INSTRUCTED_REGRESSION = 0.0005
MAX_VARIED_REGRESSION = 0.002
MIN_LIARS_GAIN = 0.02
VALIDATION_TIE = 0.0005
LIARS_TIE = 0.002


def validation_metrics(path: Path) -> dict[str, float]:
    result = json.loads(path.read_text())
    metrics = result["score_metrics"]["direct_margin"]
    return {
        "macro_auroc": float(metrics["all"]["auroc"]),
        "instructed_auroc": float(metrics["instructed"]["auroc"]),
        "varied_auroc": float(metrics["varied"]["auroc"]),
    }


def choose(
    anchor: dict[str, float],
    candidates: dict[str, dict[str, float]],
    liars: dict[str, float],
) -> dict[str, Any]:
    """Return all gate deltas and the frozen selected condition."""
    if "anchor" not in liars:
        raise ValueError("Liars results must contain an 'anchor' condition")
    rows: dict[str, Any] = {}
    eligible = []
    for name, metrics in candidates.items():
        if name not in liars:
            raise ValueError(f"Liars results are missing candidate {name!r}")
        deltas = {
            "macro_auroc": metrics["macro_auroc"] - anchor["macro_auroc"],
            "instructed_auroc": (
                metrics["instructed_auroc"] - anchor["instructed_auroc"]
            ),
            "varied_auroc": metrics["varied_auroc"] - anchor["varied_auroc"],
            "liars_macro_auroc": liars[name] - liars["anchor"],
        }
        gates = {
            "competition_macro": deltas["macro_auroc"] >= -MAX_MACRO_REGRESSION,
            "instructed": (
                deltas["instructed_auroc"] >= -MAX_INSTRUCTED_REGRESSION
            ),
            "varied": deltas["varied_auroc"] >= -MAX_VARIED_REGRESSION,
            "liars_transfer": deltas["liars_macro_auroc"] >= MIN_LIARS_GAIN,
        }
        admitted = all(gates.values())
        rows[name] = {
            "validation": metrics,
            "liars_macro_auroc": liars[name],
            "deltas_vs_anchor": deltas,
            "gates": gates,
            "eligible": admitted,
        }
        if admitted:
            eligible.append(name)

    selected = None
    rationale = "no candidate passed every frozen gate"
    if eligible:
        best_validation = max(
            candidates[name]["macro_auroc"] for name in eligible
        )
        finalists = [
            name
            for name in eligible
            if best_validation - candidates[name]["macro_auroc"]
            <= VALIDATION_TIE
        ]
        if len(finalists) == 1:
            selected = finalists[0]
            rationale = "highest competition validation macro AUROC"
        else:
            best_liars = max(liars[name] for name in finalists)
            liars_finalists = [
                name
                for name in finalists
                if best_liars - liars[name] <= LIARS_TIE
            ]
            if len(liars_finalists) == 1:
                selected = liars_finalists[0]
                rationale = "validation tie broken by Liars pilot macro AUROC"
            elif "half" in liars_finalists:
                selected = "half"
                rationale = "validation/Liars tie broken by lower OOD exposure"
            else:
                selected = sorted(liars_finalists)[0]
                rationale = "deterministic lexical tie-break"
    return {
        "thresholds": {
            "max_macro_regression": MAX_MACRO_REGRESSION,
            "max_instructed_regression": MAX_INSTRUCTED_REGRESSION,
            "max_varied_regression": MAX_VARIED_REGRESSION,
            "min_liars_gain": MIN_LIARS_GAIN,
            "validation_tie": VALIDATION_TIE,
            "liars_tie": LIARS_TIE,
        },
        "anchor": {
            "validation": anchor,
            "liars_macro_auroc": liars["anchor"],
        },
        "candidates": rows,
        "selected": selected,
        "rationale": rationale,
    }


def parse_named_method(value: str) -> tuple[str, str]:
    name, separator, method = value.partition("=")
    if not separator or not name or not method:
        raise ValueError(f"invalid condition {value!r}; expected NAME=METHOD")
    return name, method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-method", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--liars-result", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/blackbox"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    named_methods = dict(parse_named_method(value) for value in args.candidate)
    anchor = validation_metrics(
        args.results_root
        / args.anchor_method
        / args.validation_run
        / "result.json"
    )
    candidates = {
        name: validation_metrics(
            args.results_root / method / args.validation_run / "result.json"
        )
        for name, method in named_methods.items()
    }
    liars_result = json.loads(args.liars_result.read_text())
    liars = {
        name: float(condition["metrics"]["all"]["auroc"])
        for name, condition in liars_result["conditions"].items()
    }
    result = choose(anchor, candidates, liars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    if result["selected"] is None:
        raise SystemExit("no Phoenix 8.1 candidate passed the frozen gate")


if __name__ == "__main__":
    main()
