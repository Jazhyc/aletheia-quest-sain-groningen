#!/usr/bin/env python3
"""Evaluate generic Q397 generated-reasoning transfer on Liars' Bench."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_signature.evaluate_signatures import (  # noqa: E402
    CATEGORIES,
    stratified_sample,
)
from experiments.phoenix_adapter_reasoning_ndif import run as reasoning_ndif  # noqa: E402
from experiments.phoenix_adapter_validation_ndif.run import (  # noqa: E402
    MODEL_ID,
    binary_token_ids,
    hosted_metadata,
    load_credentials,
)
from experiments.phoenix_renderer_caps.run import (  # noqa: E402
    LIARS_BENCH_ID,
    LIARS_BENCH_REVISION,
)
from experiments.q397_reasoning_router.analyze import logit, sigmoid  # noqa: E402
from experiments.q397_reasoning_router.run_ndif import ADAPTER  # noqa: E402
from submission.phoenix_wright import prefix_before_prediction  # noqa: E402


METHOD = "q397_reasoning_ood_transfer_ndif_v1"
DEFAULT_FROZEN_SAMPLE = (
    ROOT
    / "results/blackbox/liars_bench_frozen_judge_signatures_v1/details.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/blackbox" / METHOD
PROMPT_DIR = ROOT / "experiments/q397_reasoning_router/prompts"
ROWS_PER_CATEGORY = 100
SAMPLE_SEED = 20260729
MAX_NEW_TOKENS = 192
DIRECT_WEIGHT = 0.6
REASONING_WEIGHT = 0.4
BOOTSTRAP_SAMPLES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-sample",
        type=Path,
        default=DEFAULT_FROZEN_SAMPLE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--generation-batches-per-session",
        type=int,
        default=32,
    )
    parser.add_argument("--score-batches-per-session", type=int, default=0)
    return parser.parse_args()


def prompt_template(name: str) -> str:
    config = yaml.safe_load((PROMPT_DIR / f"{name}.yaml").read_text())
    return str(config["student"]["prompt"])


def _category_parquet(category: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=LIARS_BENCH_ID,
        filename=f"{category}/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=LIARS_BENCH_REVISION,
    ))


def load_records(frozen_sample: Path) -> pd.DataFrame:
    frozen = pd.read_json(frozen_sample, lines=True)
    expected_categories = set(CATEGORIES)
    if set(frozen["category"]) != expected_categories:
        raise RuntimeError("frozen sample category contract changed")
    counts = frozen.groupby("category").size().to_dict()
    if any(counts.get(category) != 400 for category in CATEGORIES):
        raise RuntimeError(f"expected 400 frozen rows per category, got {counts}")

    sampled = []
    for offset, category in enumerate(CATEGORIES):
        category_rows = frozen[frozen["category"] == category].copy()
        sampled.append(
            stratified_sample(
                category_rows,
                ROWS_PER_CATEGORY,
                SAMPLE_SEED + offset,
            )
        )
    selected = pd.concat(sampled, ignore_index=True)

    records = []
    for category in CATEGORIES:
        raw = pd.read_parquet(_category_parquet(category))
        index_column = "index" if "index" in raw else "Unnamed: 0"
        raw_by_index = {
            (str(index), str(row["model"])): row
            for index, (_, row) in zip(
                raw[index_column],
                raw.iterrows(),
                strict=True,
            )
        }
        for row in selected[selected["category"] == category].itertuples():
            raw_row = raw_by_index.get(
                (str(row.row_index), str(row.source_model))
            )
            if raw_row is None:
                raise RuntimeError(
                    f"missing frozen row {category}:{row.source_model}:"
                    f"{row.row_index}"
                )
            label_column = (
                "deceptive" if "deceptive" in raw_row else "dishonest"
            )
            if int(raw_row[label_column]) != int(row.label):
                raise RuntimeError(
                    f"label mismatch for {category}:{row.row_index}"
                )
            records.append({
                "dataset": f"liars-bench/{category}",
                "category": category,
                "index": row.row_index,
                "source_model": str(row.source_model),
                "label": int(row.label),
                "messages": raw_row["messages"],
            })
    frame = pd.DataFrame(records)
    if len(frame) != ROWS_PER_CATEGORY * len(CATEGORIES):
        raise RuntimeError(f"expected 800 rows, found {len(frame)}")
    return frame


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def summarize_condition(
    records: pd.DataFrame,
    scores: np.ndarray,
) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    valid_aurocs = []
    for category, group in records.groupby("category", sort=True):
        positions = group.index.to_numpy()
        labels = group["label"].to_numpy(dtype=int)
        values = scores[positions]
        category_auroc = auroc(labels, values)
        if category_auroc is not None:
            valid_aurocs.append(category_auroc)
        categories[str(category)] = {
            "rows": len(group),
            "labels": dict(Counter(group["label"].astype(str))),
            "auroc": category_auroc,
            "unique_scores": int(len(np.unique(values))),
            "mean_score": float(np.mean(values)),
            "p95_score": float(np.quantile(values, 0.95)),
            "fraction_at_least_0_5": float(np.mean(values >= 0.5)),
        }
    return {
        "macro_auroc": float(np.mean(valid_aurocs)),
        "auroc_categories": len(valid_aurocs),
        "unique_scores": int(len(np.unique(scores))),
        "categories": categories,
    }


def compare_conditions(
    direct: dict[str, Any],
    blend: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        category: (
            blend["categories"][category]["auroc"]
            - direct["categories"][category]["auroc"]
        )
        for category in direct["categories"]
        if direct["categories"][category]["auroc"] is not None
    }
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    rng = np.random.default_rng(SAMPLE_SEED)
    draws = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    tolerance = 1e-12
    return {
        "macro_auroc_delta": (
            blend["macro_auroc"] - direct["macro_auroc"]
        ),
        "category_wins": int(np.sum(values > tolerance)),
        "category_ties": int(np.sum(np.abs(values) <= tolerance)),
        "category_losses": int(np.sum(values < -tolerance)),
        "worst_category_delta": float(np.min(values)),
        "best_category_delta": float(np.max(values)),
        "per_category_delta": deltas,
        "category_bootstrap_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def main() -> None:
    args = parse_args()
    load_credentials()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.frozen_sample.resolve())

    from nnsight import LanguageModel

    metadata = hosted_metadata(ADAPTER.repo_id)
    print(
        f"starting rows={len(records)} adapter={ADAPTER.name} "
        f"revision={metadata['revision']}",
        flush=True,
    )
    model = LanguageModel(MODEL_ID, peft=ADAPTER.repo_id)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = binary_token_ids(tokenizer)

    generation_prompts = reasoning_ndif.render_generation_prompts(
        records,
        tokenizer,
        prompt_template("summary_baseline"),
    )
    reasoning_ndif.MAX_NEW_TOKENS = MAX_NEW_TOKENS
    replies, generation_audit = reasoning_ndif.generate_reasoning(
        model,
        tokenizer,
        generation_prompts,
        batches_per_session=args.generation_batches_per_session,
    )
    direct_stems = reasoning_ndif.render_generation_prompts(
        records,
        tokenizer,
        prompt_template("binary"),
    )
    prompt_sets = {
        "direct": [prompt + "Prediction:" for prompt in direct_stems],
        "reasoning": [
            prompt + prefix_before_prediction(reply)
            for prompt, reply in zip(
                generation_prompts,
                replies,
                strict=True,
            )
        ],
    }
    scores, scoring_audit = reasoning_ndif.score_prompt_sets(
        model,
        tokenizer,
        label_ids,
        prompt_sets,
        batches_per_session=args.score_batches_per_session,
    )
    blend = sigmoid(
        DIRECT_WEIGHT * logit(scores["direct"])
        + REASONING_WEIGHT * logit(scores["reasoning"])
    )
    summaries = {
        "direct": summarize_condition(records, scores["direct"]),
        "reasoning": summarize_condition(records, scores["reasoning"]),
        "blend": summarize_condition(records, blend),
    }
    report = {
        "method": METHOD,
        "rows": len(records),
        "categories": len(CATEGORIES),
        "adapter": {
            "name": ADAPTER.name,
            "repo_id": ADAPTER.repo_id,
            "hosted": metadata,
        },
        "protocol": {
            "rows_per_category": ROWS_PER_CATEGORY,
            "sample_seed": SAMPLE_SEED,
            "frozen_sample": str(args.frozen_sample.resolve()),
            "max_new_tokens": MAX_NEW_TOKENS,
            "direct_weight": DIRECT_WEIGHT,
            "reasoning_weight": REASONING_WEIGHT,
            "specialist_prompts_used": False,
            "category_router_used": False,
            "source_reasoning_used": False,
        },
        "generation": generation_audit,
        "scoring": scoring_audit,
        "conditions": summaries,
        "blend_vs_direct": compare_conditions(
            summaries["direct"],
            summaries["blend"],
        ),
    }
    predictions = records.drop(columns="messages").copy()
    predictions["direct_score"] = scores["direct"]
    predictions["reasoning_score"] = scores["reasoning"]
    predictions["blend_score"] = blend
    predictions["reasoning_reply"] = replies
    predictions.to_json(
        output_dir / "scores.jsonl",
        orient="records",
        lines=True,
        double_precision=15,
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
