#!/usr/bin/env python3
"""Execute Phoenix 4.0 on NDIF for one reproducible batch-size condition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "submission" / "phoenix_wright_v4_0.ipynb"
SUBMISSION_CSV = ROOT / "submission.csv"
DEFAULT_DATASET = "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--medium-batch-size", type=int)
    parser.add_argument("--medium-prompt-threshold", type=int, default=1300)
    parser.add_argument("--long-batch-size", type=int)
    parser.add_argument("--long-prompt-threshold", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 1:
        raise ValueError("batch size and limit must be positive")
    if not os.environ.get("NDIF_API_KEY"):
        raise RuntimeError("NDIF_API_KEY is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    medium_batch_size = args.medium_batch_size or args.batch_size
    long_batch_size = args.long_batch_size or medium_batch_size
    os.environ.update({
        "NDIF_HOST": "https://aletheias.api.ndif.us",
        "DATASET_NAME": args.dataset,
        "ALETHEIA_LIMIT": str(args.limit),
        "PHOENIX_BATCH_SIZE": str(args.batch_size),
        "PHOENIX_MEDIUM_BATCH_SIZE": str(medium_batch_size),
        "PHOENIX_MEDIUM_PROMPT_THRESHOLD": str(args.medium_prompt_threshold),
        "PHOENIX_LONG_BATCH_SIZE": str(long_batch_size),
        "PHOENIX_LONG_PROMPT_THRESHOLD": str(args.long_prompt_threshold),
        "PHOENIX_REMOTE_BATCHES_PER_SESSION": "0",
    })
    os.environ.pop("PHOENIX_NNSIGHT_LOCAL", None)
    if SUBMISSION_CSV.exists():
        SUBMISSION_CSV.unlink()

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    started = time.perf_counter()
    executed = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    elapsed = time.perf_counter() - started
    if not SUBMISSION_CSV.exists():
        raise RuntimeError("notebook did not write submission.csv")
    frame = pd.read_csv(SUBMISSION_CSV)
    if len(frame) != args.limit:
        raise RuntimeError(f"received {len(frame)} scores, expected {args.limit}")
    if not frame["score"].between(0.0, 1.0).all():
        raise RuntimeError("notebook emitted a score outside [0, 1]")

    scores = frame["score"].to_numpy(dtype=float)
    result = {
        "dataset": args.dataset,
        "rows": len(frame),
        "batch_size": args.batch_size,
        "medium_batch_size": medium_batch_size,
        "medium_prompt_threshold": args.medium_prompt_threshold,
        "long_batch_size": long_batch_size,
        "long_prompt_threshold": args.long_prompt_threshold,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(frame) / elapsed,
        "unique_scores": int(np.unique(scores).size),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "positive_rows_at_0.15": int(frame["deceptive"].sum()),
    }
    frame.to_csv(args.output_dir / "submission.csv", index=False)
    nbformat.write(executed, args.output_dir / "executed.ipynb")
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    SUBMISSION_CSV.unlink()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
