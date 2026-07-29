#!/usr/bin/env python3
"""Summarize direct-margin metrics for the Qwen-397B distillation sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results" / "blackbox"
METHOD_RE = re.compile(
    r"^qwen9b_qwen397_tvg_soft_r(?P<rank>\d+)_lr(?P<lr_name>[^_]+)_"
    r"(?P<epoch_name>ep(?:05|1|2))_v1$"
)


def load_row(method: str, run_name: str) -> dict[str, object]:
    match = METHOD_RE.fullmatch(method)
    if match is None:
        raise ValueError(f"unexpected sweep method name: {method}")
    result_path = RESULTS_ROOT / method / run_name / "result.json"
    result = json.loads(result_path.read_text())
    config_path = RESULTS_ROOT / method / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    metrics = result["score_metrics"]["direct_margin"]
    generations_path = result_path.with_name("generations.jsonl")
    scores = {
        float(json.loads(line)["direct_margin_score"])
        for line in generations_path.read_text().splitlines()
        if line.strip()
    }
    config_text = config_path.read_text()
    epoch_match = re.search(r"^\s+num_train_epochs:\s+([0-9.]+)\s*$", config_text, re.M)
    if epoch_match is None:
        raise ValueError(f"missing num_train_epochs in {config_path}")
    effect = result.get("lora_effect")
    if not effect:
        raise ValueError(f"missing LoRA effect fingerprint in {result_path}")
    return {
        "method": method,
        "rank": int(match.group("rank")),
        "lr": float(result["learning_rate"]),
        "lr_name": match.group("lr_name"),
        "epochs": float(epoch_match.group(1)),
        "epoch_name": match.group("epoch_name"),
        "macro_auroc": float(metrics["all"]["auroc"]),
        "instructed_auroc": float(metrics["instructed"]["auroc"]),
        "varied_auroc": float(metrics["varied"]["auroc"]),
        "balanced_accuracy": float(metrics["all"]["balanced_accuracy"]),
        "unique_scores": len(scores),
        "lora_mean_abs_delta": float(effect["mean_absolute_difference"]),
    }


def markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| method | rank | LR | epochs | macro AUROC | instructed | varied | BA | unique |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['rank']} | {row['lr']:.0e} | "
            f"{row['epochs']:g} | {row['macro_auroc']:.6f} | "
            f"{row['instructed_auroc']:.6f} | {row['varied_auroc']:.6f} | "
            f"{row['balanced_accuracy']:.6f} | {row['unique_scores']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--run-name", default="validation_hparam_sweep")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--best-shell", action="store_true")
    args = parser.parse_args()

    rows = sorted(
        (load_row(method, args.run_name) for method in args.method),
        key=lambda row: (-row["macro_auroc"], -row["varied_auroc"], row["method"]),
    )
    best = rows[0]
    if args.best_shell:
        print(
            best["lr"],
            best["lr_name"],
            best["epochs"],
            best["epoch_name"],
        )
        return

    print(markdown(rows), end="")
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        (output_dir / "summary.md").write_text(markdown(rows))


if __name__ == "__main__":
    main()
