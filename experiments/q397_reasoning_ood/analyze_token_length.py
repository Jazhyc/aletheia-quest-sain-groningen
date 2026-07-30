#!/usr/bin/env python3
"""Relate token length to Q397 reasoning gains on the frozen OOD sample."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import yaml


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/blackbox/q397_reasoning_ood_transfer_ndif_v1"
DEFAULT_DIRECT = RESULTS / "structural_direct_scores.jsonl"
DEFAULT_REASONING = RESULTS / "structural_reasoning_scores.jsonl"
DEFAULT_OUTPUT = RESULTS / "token_length_analysis.json"
LIARS_ROOT = Path(
    "/scratch/s4626451/.huggingface/hub/"
    "datasets--Cadenza-Labs--liars-bench/snapshots/"
    "65299c5b10aa07adf75716ecb875c6713eed0dde"
)
TOKENIZER_JSON = Path(
    "/scratch/s4626451/.huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer.json"
)
SUMMARY_CONFIG = (
    ROOT
    / "experiments/q397_reasoning_router/prompts/summary_baseline.yaml"
)
TRUNCATION_MARKER = "\n\n[...truncated...]\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--reasoning", type=Path, default=DEFAULT_REASONING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--liars-root", type=Path, default=LIARS_ROOT)
    parser.add_argument("--tokenizer-json", type=Path, default=TOKENIZER_JSON)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_text(value: Any) -> str:
    return str(value).encode("utf-8", errors="replace").decode(
        "utf-8",
        errors="replace",
    )


def iter_messages(messages: Any) -> list[Any]:
    from collections.abc import Iterable as IterableABC
    from collections.abc import Mapping

    if messages is None:
        return []
    if isinstance(messages, Mapping):
        return [messages]
    if isinstance(messages, (str, bytes)):
        return [{"role": "", "content": messages}]
    if isinstance(messages, IterableABC):
        return list(messages)
    return [{"role": "", "content": messages}]


def raw_message_field(message: Any, field: str, default: str = "") -> str:
    from collections.abc import Mapping

    if isinstance(message, Mapping):
        value = message.get(field, default)
    elif field == "content":
        value = message
    else:
        value = default
    return safe_text(default if value is None else value)


def truncate_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - len(TRUNCATION_MARKER)
    return TRUNCATION_MARKER + text[-keep:]


def truncate_head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    available = max_chars - len(TRUNCATION_MARKER)
    head_chars = max(1, available // 3)
    tail_chars = available - head_chars
    return text[:head_chars] + TRUNCATION_MARKER + text[-tail_chars:]


def phoenix_structural_evidence(messages: Any) -> str:
    normalized = iter_messages(messages)
    output_raw = next(
        (
            raw_message_field(message, "content")
            for message in reversed(normalized)
            if raw_message_field(message, "role") == "assistant"
        ),
        "",
    )
    context_after_message_caps = "\n\n".join(
        f"{truncate_head_tail(raw_message_field(message, 'role'), 80).upper()}: "
        f"{truncate_head_tail(raw_message_field(message, 'content'), 1_500)}"
        for message in normalized
    )
    context = truncate_tail(context_after_message_caps, 3_000)
    output = truncate_head_tail(output_raw, 3_000)
    return f"\n{context}\n</context>\n\n<output>\n{output}\n</output>"


def exact_generation_prompt(member_prompt: str) -> str:
    """Apply Qwen3.5's one-user-message, no-thinking chat template."""

    return (
        "<|im_start|>user\n"
        + member_prompt.strip()
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["index"]), str(row["model"])


def score_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["category"]),
        str(row["index"]),
        str(row["source_model"]),
    )


def load_messages(
    liars_root: Path,
    wanted: dict[str, set[tuple[str, str]]],
) -> dict[tuple[str, str, str], Any]:
    messages: dict[tuple[str, str, str], Any] = {}
    for category, category_keys in wanted.items():
        path = liars_root / category / "test-00000-of-00001.parquet"
        schema_names = pq.read_schema(path).names
        index_column = "index" if "index" in schema_names else "Unnamed: 0"
        table = pq.read_table(
            path,
            columns=[index_column, "model", "messages"],
        )
        for row in table.to_pylist():
            row["index"] = row[index_column]
            if key(row) in category_keys:
                messages[(category, *key(row))] = row["messages"]
    return messages


def logit(value: float) -> float:
    clipped = min(max(float(value), 1e-8), 1.0 - 1e-8)
    return math.log(clipped) - math.log1p(-clipped)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlations(
    left: Iterable[float],
    right: Iterable[float],
) -> dict[str, float]:
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    return {
        "pearson": pearson(x, y),
        "spearman": pearson(average_ranks(x), average_ranks(y)),
    }


def auroc(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    positive = s[y == 1]
    negative = s[y == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    greater = positive[:, None] > negative[None, :]
    equal = positive[:, None] == negative[None, :]
    return float(greater.mean() + 0.5 * equal.mean())


def comparison(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left > right).astype(float) + 0.5 * (left == right)


def add_auroc_attribution(group: list[dict[str, Any]]) -> None:
    """Attribute each row its mean change in positive-negative pair ordering."""

    labels = np.asarray([row["label"] for row in group], dtype=int)
    if len(np.unique(labels)) < 2:
        return
    direct = np.asarray([row["direct_score"] for row in group], dtype=float)
    blend = np.asarray([row["blend_score"] for row in group], dtype=float)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    for index, row in enumerate(group):
        if labels[index] == 1:
            direct_pairs = comparison(direct[index], direct[negatives])
            blend_pairs = comparison(blend[index], blend[negatives])
        else:
            direct_pairs = comparison(direct[positives], direct[index])
            blend_pairs = comparison(blend[positives], blend[index])
        row["auroc_pair_gain"] = float(np.mean(blend_pairs - direct_pairs))


def paired_bootstrap_delta(
    group: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray([row["label"] for row in group], dtype=int)
    direct = np.asarray([row["direct_score"] for row in group], dtype=float)
    blend = np.asarray([row["blend_score"] for row in group], dtype=float)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    positive_draws = rng.choice(
        positives,
        size=(samples, len(positives)),
        replace=True,
    )
    negative_draws = rng.choice(
        negatives,
        size=(samples, len(negatives)),
        replace=True,
    )

    def sampled_aurocs(scores: np.ndarray) -> np.ndarray:
        positive_scores = scores[positive_draws]
        negative_scores = scores[negative_draws]
        greater = positive_scores[:, :, None] > negative_scores[:, None, :]
        equal = positive_scores[:, :, None] == negative_scores[:, None, :]
        return greater.mean(axis=(1, 2)) + 0.5 * equal.mean(axis=(1, 2))

    return sampled_aurocs(blend) - sampled_aurocs(direct)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    for group in categories.values():
        add_auroc_attribution(group)

    category_results: dict[str, Any] = {}
    for category, group in sorted(categories.items()):
        labels = [row["label"] for row in group]
        direct = [row["direct_score"] for row in group]
        blend = [row["blend_score"] for row in group]
        direct_auc = auroc(labels, direct)
        blend_auc = auroc(labels, blend)
        category_results[category] = {
            "rows": len(group),
            "median_input_tokens": float(
                np.median([row["input_tokens"] for row in group])
            ),
            "median_reasoning_tokens": float(
                np.median([row["reasoning_tokens"] for row in group])
            ),
            "direct_auroc": direct_auc,
            "blend_auroc": blend_auc,
            "blend_delta": (
                blend_auc - direct_auc
                if not math.isnan(direct_auc)
                else None
            ),
            "input_vs_signed_blend_margin_gain": correlations(
                [row["input_tokens"] for row in group],
                [row["signed_blend_margin_gain"] for row in group],
            ),
            "reasoning_length_vs_signed_blend_margin_gain": correlations(
                [row["reasoning_tokens"] for row in group],
                [row["signed_blend_margin_gain"] for row in group],
            ),
        }

    scored_categories = [
        result
        for result in category_results.values()
        if result["blend_delta"] is not None
    ]
    category_length_correlation = correlations(
        [result["median_input_tokens"] for result in scored_categories],
        [result["blend_delta"] for result in scored_categories],
    )

    metric_rows = [row for row in rows if "auroc_pair_gain" in row]
    input_values = np.asarray([row["input_tokens"] for row in metric_rows])
    signed_values = np.asarray(
        [row["signed_blend_margin_gain"] for row in metric_rows]
    )
    pair_gain_values = np.asarray(
        [row["auroc_pair_gain"] for row in metric_rows]
    )
    reasoning_lengths = np.asarray(
        [row["reasoning_tokens"] for row in metric_rows]
    )
    category_names = np.asarray([row["category"] for row in metric_rows])
    residual_input = np.empty(len(metric_rows), dtype=float)
    residual_reasoning_length = np.empty(len(metric_rows), dtype=float)
    residual_signed_gain = np.empty(len(metric_rows), dtype=float)
    residual_pair_gain = np.empty(len(metric_rows), dtype=float)
    for category in np.unique(category_names):
        mask = category_names == category
        residual_input[mask] = input_values[mask] - input_values[mask].mean()
        residual_reasoning_length[mask] = (
            reasoning_lengths[mask] - reasoning_lengths[mask].mean()
        )
        residual_signed_gain[mask] = (
            signed_values[mask] - signed_values[mask].mean()
        )
        residual_pair_gain[mask] = (
            pair_gain_values[mask] - pair_gain_values[mask].mean()
        )

    length_strata: dict[str, Any] = {}
    strata = ("short", "medium", "long")
    assigned: dict[int, str] = {}
    for group in categories.values():
        ordered = sorted(
            enumerate(group),
            key=lambda pair: pair[1]["input_tokens"],
        )
        for rank, (local_index, row) in enumerate(ordered):
            assigned[id(row)] = strata[min(2, 3 * rank // len(group))]
    for stratum in strata:
        per_category: list[dict[str, Any]] = []
        bootstrap_draws: list[np.ndarray] = []
        for category, group in sorted(categories.items()):
            selected = [row for row in group if assigned[id(row)] == stratum]
            direct_auc = auroc(
                [row["label"] for row in selected],
                [row["direct_score"] for row in selected],
            )
            blend_auc = auroc(
                [row["label"] for row in selected],
                [row["blend_score"] for row in selected],
            )
            if math.isnan(direct_auc):
                continue
            bootstrap_draws.append(
                paired_bootstrap_delta(
                    selected,
                    samples=5_000,
                    seed=20260729 + 100 * strata.index(stratum)
                    + len(per_category),
                )
            )
            per_category.append({
                "category": category,
                "rows": len(selected),
                "direct_auroc": direct_auc,
                "blend_auroc": blend_auc,
                "delta": blend_auc - direct_auc,
            })
        macro_draws = np.stack(bootstrap_draws).mean(axis=0)
        category_rng = np.random.default_rng(
            20260729 + 1_000 + strata.index(stratum)
        )
        point_deltas = np.asarray(
            [item["delta"] for item in per_category],
            dtype=float,
        )
        category_draws = category_rng.choice(
            point_deltas,
            size=(5_000, len(point_deltas)),
            replace=True,
        ).mean(axis=1)
        length_strata[stratum] = {
            "category_units": len(per_category),
            "direct_macro_auroc": float(
                np.mean([item["direct_auroc"] for item in per_category])
            ),
            "blend_macro_auroc": float(
                np.mean([item["blend_auroc"] for item in per_category])
            ),
            "delta": float(
                np.mean([item["delta"] for item in per_category])
            ),
            "delta_95_interval": [
                float(np.quantile(macro_draws, 0.025)),
                float(np.quantile(macro_draws, 0.975)),
            ],
            "probability_positive": float(np.mean(macro_draws > 0)),
            "category_resampling_95_interval": [
                float(np.quantile(category_draws, 0.025)),
                float(np.quantile(category_draws, 0.975)),
            ],
            "category_resampling_probability_positive": float(
                np.mean(category_draws > 0)
            ),
            "per_category": per_category,
        }

    return {
        "rows": len(rows),
        "rows_in_auroc_categories": len(metric_rows),
        "input_tokens": {
            "min": int(min(row["input_tokens"] for row in rows)),
            "median": float(
                np.median([row["input_tokens"] for row in rows])
            ),
            "p95": float(
                np.quantile([row["input_tokens"] for row in rows], 0.95)
            ),
            "max": int(max(row["input_tokens"] for row in rows)),
        },
        "overall_input_vs_signed_blend_margin_gain": correlations(
            input_values,
            signed_values,
        ),
        "within_category_residual_input_vs_signed_blend_margin_gain": (
            correlations(residual_input, residual_signed_gain)
        ),
        "overall_input_vs_auroc_pair_gain": correlations(
            input_values,
            pair_gain_values,
        ),
        "within_category_residual_input_vs_auroc_pair_gain": correlations(
            residual_input,
            residual_pair_gain,
        ),
        "overall_reasoning_length_vs_signed_blend_margin_gain": correlations(
            reasoning_lengths,
            signed_values,
        ),
        "within_category_residual_reasoning_length_vs_signed_gain": (
            correlations(residual_reasoning_length, residual_signed_gain)
        ),
        "overall_reasoning_length_vs_auroc_pair_gain": correlations(
            reasoning_lengths,
            pair_gain_values,
        ),
        "within_category_residual_reasoning_length_vs_auroc_pair_gain": (
            correlations(residual_reasoning_length, residual_pair_gain)
        ),
        "category_median_input_vs_category_auroc_delta": (
            category_length_correlation
        ),
        "within_category_length_terciles": length_strata,
        "categories": category_results,
    }


def main() -> None:
    args = parse_args()
    direct_rows = read_jsonl(args.direct)
    reasoning_rows = {
        score_key(row): row for row in read_jsonl(args.reasoning)
    }
    wanted: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in direct_rows:
        wanted[str(row["category"])].add(
            (str(row["index"]), str(row["source_model"]))
        )
    messages = load_messages(args.liars_root, wanted)
    if len(messages) != len(direct_rows):
        raise RuntimeError(
            f"loaded {len(messages)} messages for {len(direct_rows)} rows"
        )

    summary_prompt = yaml.safe_load(SUMMARY_CONFIG.read_text())["student"][
        "prompt"
    ]
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    rows: list[dict[str, Any]] = []
    for direct in direct_rows:
        row_key = score_key(direct)
        reasoning = reasoning_rows[row_key]
        member = (
            summary_prompt
            + "\n\n<context>"
            + phoenix_structural_evidence(messages[row_key])
        )
        prompt = exact_generation_prompt(member)
        input_tokens = len(tokenizer.encode(prompt).ids)
        reasoning_tokens = len(
            tokenizer.encode(reasoning["structural_reasoning_reply"]).ids
        )
        direct_score = float(direct["structural_direct_score"])
        reasoning_score = float(reasoning["structural_reasoning_score"])
        direct_margin = logit(direct_score)
        reasoning_margin = logit(reasoning_score)
        blend_margin = 0.6 * direct_margin + 0.4 * reasoning_margin
        blend_score = 1.0 / (1.0 + math.exp(-blend_margin))
        label = int(direct["label"])
        label_sign = 2 * label - 1
        rows.append({
            "category": str(direct["category"]),
            "index": direct["index"],
            "source_model": str(direct["source_model"]),
            "label": label,
            "input_tokens": input_tokens,
            "reasoning_tokens": reasoning_tokens,
            "direct_score": direct_score,
            "reasoning_score": reasoning_score,
            "blend_score": blend_score,
            "signed_reasoning_margin_gain": (
                label_sign * (reasoning_margin - direct_margin)
            ),
            "signed_blend_margin_gain": (
                label_sign * (blend_margin - direct_margin)
            ),
        })

    analysis = summarize_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
