#!/usr/bin/env python3
"""Pilot privileged Truth Value Guard traces from GPT-5.6 Luna on OpenRouter.

The pilot uses identical stratified varied-deception rows for every requested
reasoning effort. Responses are checkpointed immediately so interrupted runs
can resume without paying for completed requests again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    format_student_target,
    parse_teacher_target,
)
from experiments.privileged_information_distillation.generate_teacher_data import (
    load_teacher_rows,
)


load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_METHOD = "gpt56_luna_openrouter_privileged_tvg_reasoning_pilot_v1"
DEFAULT_CONFIG = ROOT / "configs" / "privileged_information_distillation.yaml"
VALID_EFFORTS = ("low", "medium", "high")


class OpenRouterError(RuntimeError):
    """An OpenRouter request or response was unusable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--efforts",
        nargs="+",
        choices=VALID_EFFORTS,
        default=["medium", "high"],
    )
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="Use these exact globally unique training indices instead of sampling.",
    )
    parser.add_argument("--seed", type=int, default=56)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--provider-sort",
        choices=("price", "throughput", "latency"),
        default="price",
    )
    parser.add_argument("--provider-only")
    parser.add_argument(
        "--allow-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "blackbox",
    )
    return parser.parse_args()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def select_stratified_rows(
    rows: list[dict[str, Any]],
    *,
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample the same number from every dataset/label stratum."""
    if per_stratum < 1:
        raise ValueError("--per-stratum must be positive")
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["label"]))].append(row)
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for stratum in sorted(grouped):
        candidates = list(grouped[stratum])
        rng.shuffle(candidates)
        if len(candidates) < per_stratum:
            raise RuntimeError(
                f"stratum {stratum!r} has {len(candidates)} rows, "
                f"need {per_stratum}"
            )
        selected.extend(candidates[:per_stratum])
    return selected


def select_rows_by_indices(
    rows: list[dict[str, Any]],
    indices: list[int],
) -> list[dict[str, Any]]:
    """Select exact globally unique indices in the requested order."""
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["index"])].append(row)
    selected: list[dict[str, Any]] = []
    for index in indices:
        matches = grouped[str(index)]
        if len(matches) != 1:
            raise ValueError(
                f"requested index {index} matched {len(matches)} teacher rows"
            )
        selected.append(matches[0])
    if len({str(index) for index in indices}) != len(indices):
        raise ValueError("--indices must not contain duplicates")
    return selected


def load_pilot_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = OmegaConf.load(args.config)
    cfg.teacher.dataset_name_contains = "varied-deception"
    cfg.teacher.limit = None
    cfg.teacher.limit_per_label = None
    cfg.teacher.shard_count = 1
    cfg.teacher.shard_index = 0
    cfg.teacher.uses_ground_truth = True
    cfg.teacher.model = args.model
    rows = load_teacher_rows(cfg, ROOT)
    if args.indices:
        return select_rows_by_indices(rows, args.indices)
    return select_stratified_rows(
        rows,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )


def request_payload(
    row: dict[str, Any],
    effort: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    provider: dict[str, Any] = {
        "require_parameters": True,
        "data_collection": "deny",
        "sort": args.provider_sort,
        "allow_fallbacks": args.allow_fallbacks,
    }
    if args.provider_only:
        provider["only"] = [args.provider_only]
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": row["teacher_prompt"]}],
        "max_tokens": args.max_tokens,
        "reasoning": {"effort": effort, "exclude": True},
        "provider": provider,
    }


def numeric_usage(usage: dict[str, Any], *keys: str) -> float:
    """Read a numeric usage field across snake/camel-case response variants."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def reasoning_tokens(usage: dict[str, Any]) -> float:
    details = (
        usage.get("completion_tokens_details")
        or usage.get("completionTokensDetails")
        or usage.get("output_tokens_details")
        or {}
    )
    if not isinstance(details, dict):
        return 0.0
    return numeric_usage(details, "reasoning_tokens", "reasoningTokens")


def retry_delay_seconds(attempt: int, response: requests.Response | None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    return min(30.0, 1.5 * (2**attempt))


def generate_one(
    row: dict[str, Any],
    effort: str,
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ndif-team/aletheias-quest-competition",
        "X-Title": "Aletheia's Quest privileged TVG teacher pilot",
    }
    payload = request_payload(row, effort, args)
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        response: requests.Response | None = None
        try:
            response = requests.post(
                args.endpoint,
                headers=headers,
                json=payload,
                timeout=args.request_timeout,
            )
            if response.status_code >= 400:
                raise OpenRouterError(
                    f"HTTP {response.status_code}: {response.text[:1000]}"
                )
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise OpenRouterError("response final content was empty")
            parsed = parse_teacher_target(
                content,
                expected_prediction=int(row["label"]),
                output_format="harmony",
            )
            if parsed is None:
                summary = prediction = None
                student_target = None
            else:
                summary, prediction = parsed
                student_target = format_student_target(summary, prediction)
            explicit_prediction = "Prediction:" in content
            usage = data.get("usage") or {}
            return {
                "dataset": row["dataset"],
                "index": row["index"],
                "label": int(row["label"]),
                "effort": effort,
                "teacher_model": args.model,
                "prompt_sha256": prompt_sha256(row["teacher_prompt"]),
                "prompt_chars": len(row["teacher_prompt"]),
                "teacher_prompt": row["teacher_prompt"],
                "student_prompt": row["student_prompt"],
                "response_id": data.get("id"),
                "response_model": data.get("model"),
                "provider": data.get("provider"),
                "finish_reason": choice.get("finish_reason"),
                "raw_completion": content,
                "reasoning_summary": summary,
                "prediction": prediction,
                "student_target": student_target,
                "parse_error": parsed is None,
                "label_match": prediction == int(row["label"]),
                "explicit_prediction": explicit_prediction,
                "usage": usage,
                "reasoning_tokens": reasoning_tokens(usage),
                "latency_seconds": time.perf_counter() - started,
                "attempts": attempt + 1,
            }
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            OpenRouterError,
            requests.RequestException,
        ) as exc:
            last_error = exc
            retryable = response is None or response.status_code in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
            }
            if attempt >= args.max_retries or not retryable:
                break
            time.sleep(retry_delay_seconds(attempt, response))
    raise OpenRouterError(
        f"{row['dataset']} index={row['index']} effort={effort} failed: "
        f"{last_error}"
    )


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["dataset"]),
        str(record["index"]),
        str(record["effort"]),
    )


def load_cache(
    path: Path,
    expected: dict[tuple[str, str, str], str],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    cached: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return cached
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record_key(record)
        if key not in expected:
            continue
        if record.get("prompt_sha256") != expected[key]:
            raise ValueError(f"cached prompt mismatch at {path}:{line_number}")
        cached[key] = record
    return cached


def summarize(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    elapsed_seconds: float,
    expected_n: int,
) -> dict[str, Any]:
    by_effort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_effort[str(record["effort"])].append(record)
    effort_results: dict[str, Any] = {}
    for effort, rows in sorted(by_effort.items()):
        usage = [row.get("usage") or {} for row in rows]
        summaries = [str(row.get("reasoning_summary") or "") for row in rows]
        effort_results[effort] = {
            "n": len(rows),
            "parse_errors": sum(bool(row.get("parse_error")) for row in rows),
            "label_matches": sum(bool(row.get("label_match")) for row in rows),
            "explicit_predictions": sum(
                bool(row.get("explicit_prediction")) for row in rows
            ),
            "summary_words_mean": (
                sum(len(text.split()) for text in summaries) / len(summaries)
                if summaries
                else 0.0
            ),
            "prompt_tokens": sum(
                numeric_usage(item, "prompt_tokens", "input_tokens") for item in usage
            ),
            "completion_tokens": sum(
                numeric_usage(item, "completion_tokens", "output_tokens")
                for item in usage
            ),
            "reasoning_tokens": sum(
                float(row.get("reasoning_tokens", 0)) for row in rows
            ),
            "reported_cost": sum(numeric_usage(item, "cost") for item in usage),
            "latency_seconds_sum": sum(
                float(row.get("latency_seconds", 0)) for row in rows
            ),
            "finish_reasons": dict(
                Counter(str(row.get("finish_reason")) for row in rows)
            ),
            "providers": dict(Counter(str(row.get("provider")) for row in rows)),
            "retry_rows": sum(int(row.get("attempts", 1)) > 1 for row in rows),
        }
    return {
        "method": args.method,
        "model": args.model,
        "complete": len(records) == expected_n,
        "n": len(records),
        "elapsed_seconds": elapsed_seconds,
        "config": {
            "efforts": args.efforts,
            "per_stratum": args.per_stratum,
            "indices": args.indices,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "provider_sort": args.provider_sort,
            "provider_only": args.provider_only,
            "allow_fallbacks": args.allow_fallbacks,
            "prompt_contract": "privileged Truth Value Guard reasoning summary",
        },
        "efforts": effort_results,
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing (it may be set in .env)")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    rows = load_pilot_rows(args)
    work = [(row, effort) for effort in args.efforts for row in rows]
    expected = {
        (str(row["dataset"]), str(row["index"]), effort): prompt_sha256(
            row["teacher_prompt"]
        )
        for row, effort in work
    }
    if len(expected) != len(work):
        raise ValueError("pilot work identities were not unique")

    run_dir = args.output_root / args.method
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "generations.jsonl"
    cached = load_cache(cache_path, expected)
    pending = [
        (row, effort)
        for row, effort in work
        if (str(row["dataset"]), str(row["index"]), effort) not in cached
    ]
    print(
        f"prepared={len(work)} cached={len(cached)} pending={len(pending)} "
        f"rows={len(rows)} efforts={','.join(args.efforts)}",
        flush=True,
    )

    started = time.perf_counter()
    write_lock = threading.Lock()
    failures: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(generate_one, row, effort, args, api_key): (row, effort)
            for row, effort in pending
        }
        for future in as_completed(futures):
            row, effort = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                failure = (
                    f"{row['dataset']} index={row['index']} effort={effort}: {exc}"
                )
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)
                continue
            cached[record_key(record)] = record
            with write_lock:
                with cache_path.open("a") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
            completed += 1
            if completed <= 4 or completed % 12 == 0 or completed == len(pending):
                usage = record.get("usage") or {}
                print(
                    f"completed={completed}/{len(pending)} effort={effort} "
                    f"provider={record.get('provider')} "
                    f"prompt={numeric_usage(usage, 'prompt_tokens', 'input_tokens'):.0f} "
                    f"completion={numeric_usage(usage, 'completion_tokens', 'output_tokens'):.0f} "
                    f"reasoning={record.get('reasoning_tokens', 0):.0f} "
                    f"cost={numeric_usage(usage, 'cost'):.6f}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    ordered = [
        cached[(str(row["dataset"]), str(row["index"]), effort)]
        for row, effort in work
        if (str(row["dataset"]), str(row["index"]), effort) in cached
    ]
    result = summarize(ordered, args, elapsed, expected_n=len(work))
    result["failures"] = failures
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"wrote {result_path}", flush=True)
    if failures or not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
