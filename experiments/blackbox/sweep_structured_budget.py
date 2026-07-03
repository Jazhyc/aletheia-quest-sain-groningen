#!/usr/bin/env python3
"""Submit, run, or plot a structured-reasoning thinking-budget sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGETS = [256, 512, 1024, 2048, 4096]
METRIC_KEYS = ["balanced_accuracy", "auroc", "recall", "fpr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("submit", "plot", "run-local"), default="submit",
                        help="submit Slurm jobs, plot finished jobs, or run budgets sequentially in-process.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config-name", default="blackbox_structured_reasoning")
    parser.add_argument("--method-prefix", default="qwen_structured_reason_budget")
    parser.add_argument("--output-dir", default="results/blackbox")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--job-script", default="experiments/blackbox/run_judge.sh")
    parser.add_argument("--sbatch-arg", action="append", default=[],
                        help="Extra argument passed before the Slurm job script; repeat as needed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print submit/run commands without executing them.")
    parser.add_argument("overrides", nargs="*",
                        help="Extra Hydra overrides forwarded to run_judge.py.")
    return parser.parse_args()


def method_name(prefix: str, budget: int) -> str:
    return f"{prefix}_{budget}"


def judge_overrides(args: argparse.Namespace, budget: int, results_root: Path) -> list[str]:
    method = method_name(args.method_prefix, budget)
    return [
        "--config-name",
        args.config_name,
        f"split={args.split}",
        f"method={method}",
        f"output_dir={results_root.as_posix()}",
        f"judge.max_tokens={budget}",
        *args.overrides,
    ]


def submit_budget(args: argparse.Namespace, budget: int, results_root: Path) -> None:
    command = [
        "sbatch",
        *args.sbatch_arg,
        args.job_script,
        *judge_overrides(args, budget, results_root),
    ]
    print("submitting", " ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def run_budget(args: argparse.Namespace, budget: int, results_root: Path) -> None:
    command = [
        sys.executable,
        "experiments/blackbox/run_judge.py",
        *judge_overrides(args, budget, results_root),
    ]
    print("running", " ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def read_result(path: Path, budget: int, method: str) -> dict[str, object]:
    record = json.loads(path.read_text())
    metrics = record.get("metrics", {})
    timing = record.get("timing", {})
    row = {
        "budget": budget,
        "method": method,
        "n": record.get("n"),
        "parse_errors": record.get("parse_errors", 0),
        "score_seconds": timing.get("score_seconds"),
        "rows_per_second": timing.get("rows_per_second"),
        "result_path": path.as_posix(),
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    return row


def collect_rows(args: argparse.Namespace, results_root: Path) -> list[dict[str, object]]:
    rows = []
    for budget in args.budgets:
        method = method_name(args.method_prefix, budget)
        path = results_root / method / args.split / "result.json"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        rows.append(read_result(path, budget, method))
    rows.sort(key=lambda row: int(row["budget"]))
    if not rows:
        raise SystemExit("no result.json files found for the requested sweep")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "budget",
        "method",
        "n",
        "balanced_accuracy",
        "auroc",
        "recall",
        "fpr",
        "parse_errors",
        "score_seconds",
        "rows_per_second",
        "result_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(row: dict[str, object], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        return float("nan")
    return float(value)


def scaled(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def write_svg(path: Path, rows: list[dict[str, object]], *, metric: str = "balanced_accuracy") -> None:
    plot_rows = [row for row in rows if not math.isnan(metric_value(row, metric))]
    if not plot_rows:
        raise SystemExit(f"no numeric {metric} values available to plot")

    width, height = 900, 560
    left, right, top, bottom = 92, 36, 58, 86
    chart_w = width - left - right
    chart_h = height - top - bottom
    budgets = [int(row["budget"]) for row in plot_rows]
    x_values = [math.log2(budget) for budget in budgets]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.0, 1.0

    def x_pos(budget: int) -> float:
        return scaled(math.log2(budget), x_min, x_max, left, left + chart_w)

    def y_pos(value: float) -> float:
        return scaled(value, y_min, y_max, top + chart_h, top)

    points = [
        (x_pos(int(row["budget"])), y_pos(metric_value(row, metric)), row)
        for row in plot_rows
    ]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560">',
        '<rect width="900" height="560" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a} .tick{fill:#5d6d7e;font-size:12px} '
        '.label{font-size:15px;font-weight:600} .title{font-size:22px;font-weight:700}</style>',
        f'<text class="title" x="{width / 2:.1f}" y="32" text-anchor="middle">'
        'Structured Reasoning Budget Sweep</text>',
    ]

    for i in range(11):
        y_value = i / 10
        y = y_pos(y_value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" '
                     'stroke="#e5e8e8" stroke-width="1"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">'
                     f'{y_value:.1f}</text>')

    lines.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" '
                 f'y2="{top + chart_h}" stroke="#17202a" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" '
                 'stroke="#17202a" stroke-width="1.5"/>')

    for budget in budgets:
        x = x_pos(budget)
        lines.append(f'<line x1="{x:.1f}" y1="{top + chart_h}" x2="{x:.1f}" '
                     f'y2="{top + chart_h + 6}" stroke="#17202a" stroke-width="1.2"/>')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{top + chart_h + 24}" '
                     f'text-anchor="middle">{budget}</text>')

    lines.append(f'<polyline points="{polyline}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    for x, y, row in points:
        value = metric_value(row, metric)
        parse_errors = row.get("parse_errors", 0)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#1f77b4"/>')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle">'
                     f'{value:.3f}</text>')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{y + 24:.1f}" text-anchor="middle">'
                     f'parse={escape(str(parse_errors))}</text>')

    lines.append(f'<text class="label" x="{left + chart_w / 2:.1f}" y="{height - 28}" '
                 'text-anchor="middle">Thinking budget (tokens, log2 spacing)</text>')
    lines.append(f'<text class="label" transform="translate(24 {top + chart_h / 2:.1f}) rotate(-90)" '
                 'text-anchor="middle">Balanced accuracy</text>')
    lines.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    results_root = Path(args.output_dir)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else results_root / "structured_budget_sweep" / args.split
    if not artifacts_dir.is_absolute():
        artifacts_dir = ROOT / artifacts_dir

    if args.mode == "submit":
        for budget in args.budgets:
            submit_budget(args, budget, results_root)
        print("submitted all budget jobs")
        return

    if args.mode == "run-local":
        for budget in args.budgets:
            run_budget(args, budget, results_root)

    rows = collect_rows(args, results_root)
    csv_path = artifacts_dir / "metrics.csv"
    svg_path = artifacts_dir / "accuracy_vs_thinking_budget.svg"
    write_csv(csv_path, rows)
    write_svg(svg_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
