#!/usr/bin/env python
"""Execute a submission notebook against one public dataset.

Local NNsight execution remains the default.  ``--remote`` instead exercises the
same NDIF path used by leaderboard evaluation, which is useful for bounded
capacity checks without changing the submission notebook.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


DEFAULT_DATASET = (
    "aletheias-quest/"
    "dev-instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None"
)
DEFAULT_NDIF_HOST = "https://aletheias.api.ndif.us"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default="submission/phoenix_wright_v1_3.ipynb")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--keep-submission", action="store_true")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--ndif-host", default=DEFAULT_NDIF_HOST)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--disable-adapter", action="store_true")
    parser.add_argument(
        "--model-wrapper",
        choices=("language", "vision_language"),
    )
    return parser.parse_args()


def validate_submission(path: Path, expected_rows: int) -> None:
    if not path.exists():
        raise RuntimeError(f"{path} was not written")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{path} has no rows")
    if expected_rows and len(rows) != expected_rows:
        raise RuntimeError(f"{path} has {len(rows)} rows, expected {expected_rows}")
    expected_columns = {"index", "deceptive", "score"}
    if set(rows[0]) != expected_columns:
        raise RuntimeError(f"{path} columns are {list(rows[0])}, expected {sorted(expected_columns)}")
    for row in rows:
        score = float(row["score"])
        if not 0.0 <= score <= 1.0:
            raise RuntimeError(f"score out of range: {score}")
        if row["deceptive"] not in {"True", "False", "true", "false", "0", "1"}:
            raise RuntimeError(f"unexpected deceptive value: {row['deceptive']!r}")
    print(f"validated {path} with {len(rows)} rows")


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook)
    output_path = Path("submission.csv")
    if output_path.exists():
        output_path.unlink()

    os.environ["DATASET_NAME"] = args.dataset
    os.environ["ALETHEIA_LIMIT"] = str(args.limit)
    if args.remote:
        os.environ.pop("PHOENIX_NNSIGHT_LOCAL", None)
        os.environ["NDIF_HOST"] = args.ndif_host
    else:
        os.environ["PHOENIX_NNSIGHT_LOCAL"] = "1"
    if args.batch_size is not None:
        os.environ["PHOENIX_BATCH_SIZE"] = str(args.batch_size)
    else:
        os.environ.setdefault("PHOENIX_BATCH_SIZE", "1")
    if args.max_prompt_tokens is not None:
        os.environ["PHOENIX_MAX_PROMPT_TOKENS"] = str(args.max_prompt_tokens)
    if args.max_new_tokens is not None:
        os.environ["PHOENIX_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    if args.include_reasoning:
        os.environ["PHOENIX_INCLUDE_REASONING"] = "1"
    if args.disable_adapter:
        os.environ["PHOENIX_DISABLE_ADAPTER"] = "1"
    if args.model_wrapper is not None:
        os.environ["PHOENIX_MODEL_WRAPPER"] = args.model_wrapper

    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    print(
        f"executing {notebook_path} dataset={args.dataset} "
        f"limit={args.limit} remote={args.remote} "
        f"batch_size={os.environ['PHOENIX_BATCH_SIZE']} "
        f"max_prompt_tokens={os.environ.get('PHOENIX_MAX_PROMPT_TOKENS', 'not-set')} "
        f"max_new_tokens={os.environ.get('PHOENIX_MAX_NEW_TOKENS', 'not-set')}"
    )
    client.execute()
    validate_submission(output_path, args.limit)
    if not args.keep_submission:
        output_path.unlink()
        print(f"removed {output_path}")


if __name__ == "__main__":
    main()
