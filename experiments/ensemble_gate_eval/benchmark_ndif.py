#!/usr/bin/env python3
"""Execute one submission notebook on NDIF and score it against dev labels.

This is ``benchmark_phoenix_v4_ndif.py`` generalized to any notebook, plus the
metric the leaderboard now ranks on.  Running both notebooks through the same
harness, on the same dataset and row count, is what makes sonic v3 and Phoenix
Wright 4.0 directly comparable -- the recorded v4 numbers came from a script
that reported score spread but not AUROC.

    python experiments/ensemble_gate_eval/benchmark_ndif.py \\
        --notebook submission/sonic_v3.ipynb --output-dir results/sonic_v3/bench
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "dev_splits/manifest.csv"
SUBMISSION_CSV = ROOT / "submission.csv"
DEFAULT_DATASET = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"


def load_labels(dataset: str) -> dict[str, bool]:
    """
    Return ``index -> deceptive`` for one dev dataset.

    Prefers the local manifest, then falls back to the dataset's sibling
    ``-labels`` repo on the Hub. The fallback matters: the most useful
    benchmarks are the datasets deliberately kept OUT of
    ``dev_splits/manifest.csv`` (the two ``g-st`` sets), because no shipped
    probe has trained on them and they are therefore leakage-free.

    :param dataset: Full HF dataset id.
    :return: Mapping from stringified row index to the deceptive flag.
    :raises SystemExit: If neither source yields labels.
    """
    labels = {}
    with open(MANIFEST) as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset:
                labels[str(row["index"])] = row["deceptive"] == "True"
    if labels:
        return labels

    from datasets import load_dataset
    try:
        sibling = load_dataset(f"{dataset}-labels", split="test")
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        raise SystemExit(
            f"no manifest labels for {dataset}, and its -labels repo failed "
            f"to load: {type(error).__name__}: {error}"
        ) from error
    labels = {str(index): bool(flag)
              for index, flag in zip(sibling["index"], sibling["deceptive"])}
    if not labels:
        raise SystemExit(f"no labels found for {dataset}")
    print(f"labels: manifest miss, using {dataset}-labels ({len(labels)} rows)")
    return labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, required=True,
                        help="path to the notebook to execute (absolute or "
                             "relative to the repository root)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override PHOENIX_BATCH_SIZE for the judge")
    args = parser.parse_args(argv)

    if not os.environ.get("NDIF_API_KEY"):
        raise SystemExit("NDIF_API_KEY is required")
    notebook_path = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    if not notebook_path.exists():
        raise SystemExit(f"{notebook_path} does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.update({
        "NDIF_HOST": os.environ.get("NDIF_HOST", "https://aletheias.api.ndif.us"),
        "DATASET_NAME": args.dataset,
        "ALETHEIA_LIMIT": str(args.limit),
        "NNSIGHT_REMOTE": "1",
    })
    if args.batch_size:
        os.environ["PHOENIX_BATCH_SIZE"] = str(args.batch_size)
    os.environ.pop("PHOENIX_NNSIGHT_LOCAL", None)
    if SUBMISSION_CSV.exists():
        SUBMISSION_CSV.unlink()

    notebook = nbformat.read(notebook_path, as_version=4)
    started = time.perf_counter()
    executed = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    elapsed = time.perf_counter() - started

    # Save the executed notebook before scoring: when a run fails, its cell
    # output is the only record of why, and metric errors must not discard it.
    nbformat.write(executed, args.output_dir / "executed.ipynb")

    if not SUBMISSION_CSV.exists():
        raise SystemExit("notebook did not write submission.csv")
    frame = pd.read_csv(SUBMISSION_CSV)
    if frame.empty:
        raise SystemExit(
            f"notebook wrote an empty submission.csv -- see "
            f"{args.output_dir / 'executed.ipynb'} for the failing cell"
        )
    if not frame["score"].between(0.0, 1.0).all():
        raise SystemExit("notebook emitted a score outside [0, 1]")

    labels = load_labels(args.dataset)
    truth = np.array([labels[str(index)] for index in frame["index"]])
    scores = frame["score"].to_numpy(dtype=float)
    result = {
        "notebook": str(notebook_path.relative_to(ROOT)),
        "dataset": args.dataset,
        "rows": int(len(frame)),
        "auroc": float(roc_auc_score(truth, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, frame["deceptive"])),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(frame) / elapsed,
        "unique_scores": int(np.unique(scores).size),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "positive_rows": int(frame["deceptive"].sum()),
        "deceptive_rows": int(truth.sum()),
    }
    frame.to_csv(args.output_dir / "submission.csv", index=False)
    nbformat.write(executed, args.output_dir / "executed.ipynb")
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    SUBMISSION_CSV.unlink()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
