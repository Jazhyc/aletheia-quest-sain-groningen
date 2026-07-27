#!/usr/bin/env python3
"""Analyze persona-belief prompt members using AUROC-only selection."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "blackbox"))

from run_judge import (  # noqa: E402
    load_examples_for_labels,
    load_label_subset,
    load_split_config,
)


CONTROL = "truth_value_guard_control"
ROLEPLAY_RE = re.compile(
    r"\b(?:role[- ]?play|playing the character|fictional scenario|"
    r"parallel universe|simulation 7b|persona lock|trivia game|"
    r"play a game|inverse-gpt|wrong answer mode|lorekeeper|quiz master|"
    r"your name is atlas|polished spokesperson|reverse turing test)\b",
    flags=re.IGNORECASE,
)


def system_text(messages: Any) -> str:
    """Return all system-message text without using labels or assistant output."""
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    )


def is_persona_roleplay(messages: Any) -> bool:
    """Apply the frozen label-blind persona/roleplay diagnostic."""
    return bool(ROLEPLAY_RE.search(system_text(messages)))


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return AUROC when both classes are present."""
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def mean_available(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return float(np.mean(available)) if available else None


def summarize_member(frames: list[pd.DataFrame]) -> dict[str, Any]:
    """Summarize one prompt member without computing thresholded metrics."""
    per_dataset: dict[str, dict[str, Any]] = {}
    scenario_values: dict[str, list[float | None]] = defaultdict(list)
    all_scores: list[np.ndarray] = []
    for frame in frames:
        dataset = str(frame["dataset"].iloc[0])
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame["score"].to_numpy(dtype=float)
        auroc = safe_auroc(labels, scores)
        scenario = (
            "varied"
            if "varied-deception" in dataset
            else "instructed"
            if "instructed-deception" in dataset
            else "other"
        )
        scenario_values[scenario].append(auroc)
        per_dataset[dataset] = {
            "n": int(len(frame)),
            "auroc": auroc,
            "unique_scores": int(frame["score"].nunique(dropna=False)),
        }
        all_scores.append(scores)

    strata: dict[str, dict[str, Any]] = {}
    for name, wanted in (("persona", True), ("non_persona", False)):
        stratum_frames = [frame.loc[frame["persona"] == wanted] for frame in frames]
        stratum_frames = [frame for frame in stratum_frames if len(frame)]
        labels = np.concatenate(
            [frame["label"].to_numpy(dtype=int) for frame in stratum_frames]
        ) if stratum_frames else np.asarray([], dtype=int)
        scores = np.concatenate(
            [frame["score"].to_numpy(dtype=float) for frame in stratum_frames]
        ) if stratum_frames else np.asarray([], dtype=float)
        strata[name] = {
            "n": int(len(labels)),
            "positives": int(labels.sum()),
            "pooled_auroc": safe_auroc(labels, scores),
            "macro_available_dataset_auroc": mean_available([
                safe_auroc(
                    frame["label"].to_numpy(dtype=int),
                    frame["score"].to_numpy(dtype=float),
                )
                for frame in stratum_frames
            ]),
            "datasets_with_both_labels": int(sum(
                frame["label"].nunique() == 2 for frame in stratum_frames
            )),
        }

    scores = np.concatenate(all_scores)
    unique_scores = int(len(np.unique(scores)))
    return {
        "macro_per_dataset_auroc": mean_available([
            item["auroc"] for item in per_dataset.values()
        ]),
        "scenario_macro_auroc": {
            scenario: mean_available(values)
            for scenario, values in sorted(scenario_values.items())
        },
        "per_dataset": per_dataset,
        "strata": strata,
        "score_continuity": {
            "n": int(len(scores)),
            "unique_scores": unique_scores,
            "ties": int(len(scores) - unique_scores),
            "unique_fraction": float(unique_scores / len(scores)),
        },
    }


def resolve_result_path(pathish: str) -> Path:
    path = Path(pathish)
    return path if path.is_absolute() else ROOT / path


def load_member_frames(
    result: dict[str, Any],
    split_config: Path,
) -> dict[str, list[pd.DataFrame]]:
    configs = {
        config.name: config
        for config in load_split_config(split_config, ROOT)
    }
    by_member: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for dataset, member_paths in result["member_predictions"].items():
        config = configs[dataset]
        labels = load_label_subset(config).copy()
        labels["_key"] = labels["index"].astype(str)
        examples = load_examples_for_labels(dataset, labels, None)
        persona_by_index = {
            str(example["index"]): is_persona_roleplay(example["messages"])
            for example in examples
        }
        for member, pathish in member_paths.items():
            predictions = pd.read_csv(resolve_result_path(pathish))
            predictions["_key"] = predictions["index"].astype(str)
            frame = labels[["_key", "label"]].merge(
                predictions[["_key", "score"]],
                on="_key",
                how="inner",
                validate="one_to_one",
            )
            if len(frame) != len(labels):
                raise ValueError(
                    f"{dataset}/{member}: matched {len(frame)} of {len(labels)} labels"
                )
            frame["dataset"] = dataset
            frame["persona"] = frame["_key"].map(persona_by_index).fillna(False)
            by_member[member].append(frame)
    return by_member


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=ROOT / "dev_splits" / "dry.validation.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result_path = resolve_result_path(str(args.result_json))
    result = json.loads(result_path.read_text())
    frames = load_member_frames(result, args.split_config.resolve())
    summaries = {
        member: summarize_member(member_frames)
        for member, member_frames in frames.items()
    }
    control = summaries[CONTROL]["macro_per_dataset_auroc"]
    for summary in summaries.values():
        value = summary["macro_per_dataset_auroc"]
        summary["delta_vs_control"] = (
            None if value is None or control is None else float(value - control)
        )

    ranked = sorted(
        summaries.items(),
        key=lambda item: item[1]["macro_per_dataset_auroc"] or float("-inf"),
        reverse=True,
    )
    print("member                              macro_AUROC   delta      unique")
    for member, summary in ranked:
        continuity = summary["score_continuity"]
        print(
            f"{member:<35} "
            f"{fmt(summary['macro_per_dataset_auroc']):>11} "
            f"{fmt(summary['delta_vs_control']):>10} "
            f"{continuity['unique_scores']:>4}/{continuity['n']}"
        )
        for stratum in ("persona", "non_persona"):
            stats = summary["strata"][stratum]
            print(
                f"  {stratum:<33} n={stats['n']:<4} "
                f"pooled={fmt(stats['pooled_auroc'])} "
                f"macro_available={fmt(stats['macro_available_dataset_auroc'])}"
            )

    report = {
        "selection_metric": "mean per-dataset AUROC",
        "selection_uses_thresholded_metrics": False,
        "result_json": result_path.as_posix(),
        "split_config": args.split_config.resolve().as_posix(),
        "roleplay_definition": ROLEPLAY_RE.pattern,
        "members": summaries,
        "ranking": [member for member, _ in ranked],
    }
    output = args.output or result_path.parent / "persona_belief_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
