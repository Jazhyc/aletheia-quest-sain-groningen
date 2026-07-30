#!/usr/bin/env python3
"""Query Kimi K3 literal binary logits for arbitrary Liars' Bench JSONL rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.openrouter_qwen397_tvg.run_openrouter_tvg import (
    DEFAULT_ENDPOINT,
    WorkItem,
    load_cache,
    prompt_sha256,
    score_item,
)
from experiments.privileged_information_distillation.core import build_student_prompt
from experiments.kimi_liars_enrichment.prompts import CATEGORY_SPECS
from submission.phoenix_wright import DECEPTION_PROMPT


load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=("ordinary", "semantic", "selected"), required=True
    )
    parser.add_argument("--model", default="moonshotai/kimi-k3")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--provider-only", default="Fireworks")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return rows


def category_for(row: dict) -> str:
    category = str(
        row.get("category")
        or str(row.get("dataset", "")).removeprefix("liars-bench/")
    )
    if category not in CATEGORY_SPECS:
        raise ValueError(f"unsupported Liars' Bench category: {category!r}")
    return category


def make_work(rows: list[dict], condition: str) -> list[WorkItem]:
    work = []
    for row in rows:
        category = category_for(row)
        spec = CATEGORY_SPECS[category]
        row_condition = (
            str(row.get("teacher_prompt_kind")) if condition == "selected" else condition
        )
        if row_condition not in {"ordinary", "semantic"}:
            raise ValueError(
                f"row lacks a valid selected teacher condition: {row_condition!r}"
            )
        if row_condition == "semantic":
            template = str(spec["semantic_prompt"])
            max_chars = int(spec["teacher_max_chars"])
            truncation = str(spec["teacher_truncation"])
        else:
            template = DECEPTION_PROMPT
            max_chars = 3000
            truncation = "tail"
        prompt = build_student_prompt(
            row["messages"], template, max_chars, truncation
        )
        work.append(
            WorkItem(
                dataset=f"liars-bench/{category}",
                index=str(row["index"]),
                label=int(row["label"]),
                prompt=prompt,
                prompt_sha256=prompt_sha256(prompt),
                prompt_chars=len(prompt),
            )
        )
    return work


def runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        endpoint=args.endpoint,
        provider_sort="price",
        provider_only=args.provider_only,
        allow_fallbacks=False,
        max_tokens=8,
        top_logprobs=5,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
    )


def atomic_write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in sorted(
                records, key=lambda row: (row["dataset"], str(row["index"]))
            )
        )
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing (it may be set in .env)")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    rows = load_rows(args.input)
    work = make_work(rows, args.condition)
    by_key = {item.key: item for item in work}
    condition_by_key = {
        (f"liars-bench/{category_for(row)}", str(row["index"])): (
            str(row.get("teacher_prompt_kind"))
            if args.condition == "selected"
            else args.condition
        )
        for row in rows
    }
    if len(by_key) != len(work):
        raise ValueError("input dataset/index identities are not unique")
    cached = load_cache(args.output, by_key)
    pending = [item for item in work if item.key not in cached]
    print(
        f"condition={args.condition} prepared={len(work)} "
        f"cached={len(cached)} pending={len(pending)}",
        flush=True,
    )
    started = time.perf_counter()
    lock = threading.Lock()
    failures = []
    scoring_args = runtime_args(args)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(score_item, item, scoring_args, api_key): item
            for item in pending
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                failures.append(f"{item.dataset} index={item.index}: {exc}")
                print(f"FAILED {failures[-1]}", flush=True)
                continue
            record["condition"] = condition_by_key[item.key]
            cached[item.key] = record
            with lock:
                atomic_write(args.output, list(cached.values()))
            if len(cached) % 100 == 0:
                print(f"completed={len(cached)}/{len(work)}", flush=True)
    atomic_write(args.output, list(cached.values()))
    print(
        f"complete={len(cached) == len(work)} rows={len(cached)}/{len(work)} "
        f"failures={len(failures)} seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    if failures or len(cached) != len(work):
        raise SystemExit("Kimi query incomplete; rerun to resume")


if __name__ == "__main__":
    main()
