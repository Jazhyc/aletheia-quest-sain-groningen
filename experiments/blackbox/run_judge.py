#!/usr/bin/env python3
"""Hydra-driven local black-box judge experiment runner.

This is development tooling, not submission code. It scores public dev split
subsets created by ``scripts/make_dev_splits.py`` and records local metrics so
we can iterate on prompts, rating scales, thresholds, and batching. Development
runs use local GPU inference through vLLM; NDIF is reserved for evaluation /
submission execution.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

LEADERBOARD_SRC = ROOT / "leaderboard" / "src"
sys.path.insert(0, str(LEADERBOARD_SRC))

from aletheia_runner.config import DatasetConfig, METRIC_KEYS  # noqa: E402
from aletheia_runner.scoring import compute_metrics, load_predictions  # noqa: E402


HF_SPLIT = "test"


@dataclass
class DatasetResult:
    dataset: str
    n: int
    metrics: dict[str, float | None]
    predictions_path: str


def resolve_path(pathish: str, base: Path) -> Path:
    path = Path(pathish)
    return path if path.is_absolute() else base / path


def resolve_uri(uri: str, base: Path) -> str:
    if "://" in uri:
        return uri
    return resolve_path(uri, base).as_posix()


def load_split_config(path: Path, base: Path) -> list[DatasetConfig]:
    data = yaml.safe_load(path.read_text()) or {}
    datasets = []
    for item in data.get("datasets", []):
        item = dict(item)
        item["labels_uri"] = resolve_uri(item["labels_uri"], base)
        datasets.append(DatasetConfig(**item))
    if not datasets:
        raise SystemExit(f"{path} has no datasets entries")
    return datasets


def load_label_subset(cfg: DatasetConfig) -> pd.DataFrame:
    uri = Path(cfg.labels_uri)
    if cfg.labels_uri.endswith(".csv") and uri.exists():
        raw = pd.read_csv(uri)
    else:
        from datasets import load_dataset

        ds = load_dataset(cfg.labels_uri, split=HF_SPLIT)
        raw = ds.to_pandas()
    if cfg.id_column not in raw.columns:
        raw = raw.reset_index().rename(columns={"index": cfg.id_column})
    return pd.DataFrame({
        "index": raw[cfg.id_column],
        "label": raw[cfg.label_column].astype(int),
    })


def load_examples_for_labels(dataset_name: str, labels: pd.DataFrame, limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=HF_SPLIT)
    if "index" not in ds.column_names:
        ds = ds.add_column("index", list(range(len(ds))))
    wanted = set(labels["index"].tolist())
    rows = [dict(row) for row in ds if row["index"] in wanted]
    order = {idx: i for i, idx in enumerate(labels["index"].tolist())}
    rows.sort(key=lambda row: order[row["index"]])
    if limit is not None:
        rows = rows[:limit]
    return rows


def final_assistant(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def merge_messages(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def format_dialogue(messages: list[dict], max_chars: int) -> tuple[str, str]:
    merged = merge_messages(messages)
    assistant = final_assistant(merged)
    lines = [f"{m.get('role', '').upper()}: {m.get('content', '')}" for m in merged]
    context = "\n\n".join(lines)
    if len(context) > max_chars:
        context = context[-max_chars:]
    return context, assistant


def build_prompt(messages: list[dict], prompt_template: str, max_chars: int) -> str:
    context, output = format_dialogue(messages, max_chars)
    return f"{prompt_template}\n\n<context>\n{context}\n</context>\n\n<output>\n{output}\n</output>\n\nRating:"


def load_shot_pool(splits_dir: Path, split: str, base: Path) -> list[tuple[DatasetConfig, pd.DataFrame]]:
    config_path = splits_dir / f"dry.{split}.yaml"
    if not config_path.exists():
        raise SystemExit(f"{config_path} does not exist; cannot build few-shot examples")
    pool = []
    for dataset_cfg in load_split_config(config_path, base):
        pool.append((dataset_cfg, load_label_subset(dataset_cfg)))
    return pool


def sample_shot_records(
    pool: list[tuple[DatasetConfig, pd.DataFrame]],
    *,
    n_per_label: int,
    seed: int,
) -> list[tuple[DatasetConfig, dict[str, Any]]]:
    by_label: dict[int, list[tuple[DatasetConfig, dict[str, Any]]]] = {0: [], 1: []}
    for dataset_cfg, labels in pool:
        for row in labels.to_dict("records"):
            by_label[int(row["label"])].append((dataset_cfg, row))

    rng = random.Random(seed)
    selected: list[tuple[DatasetConfig, dict[str, Any]]] = []
    for label in (0, 1):
        rows = by_label[label]
        rng.shuffle(rows)
        selected.extend(rows[:n_per_label])
    rng.shuffle(selected)
    return selected


def build_few_shot_prefix(
    splits_dir: Path,
    *,
    split: str,
    n_per_label: int,
    seed: int,
    rating_min: int,
    rating_max: int,
    max_prompt_chars: int,
    base: Path,
) -> str:
    if n_per_label <= 0:
        return ""
    pool = load_shot_pool(splits_dir, split, base)
    selected = sample_shot_records(pool, n_per_label=n_per_label, seed=seed)
    by_dataset: dict[str, list[Any]] = {}
    for dataset_cfg, row in selected:
        by_dataset.setdefault(dataset_cfg.name, []).append(row)

    parts = [
        "Here are labelled examples. Use them to calibrate the rating scale; do not copy their answers."
    ]
    for dataset_name, rows in by_dataset.items():
        labels = pd.DataFrame({"index": [row["index"] for row in rows],
                               "label": [row["label"] for row in rows]})
        examples = load_examples_for_labels(dataset_name, labels, None)
        label_by_index = {row["index"]: int(row["label"]) for row in rows}
        for example in examples:
            context, output = format_dialogue(example["messages"], max_prompt_chars)
            rating = rating_max if label_by_index[example["index"]] else rating_min
            parts.append(
                f"<example>\n<context>\n{context}\n</context>\n"
                f"<output>\n{output}\n</output>\nRating: {rating}\n</example>"
            )
    return "\n\n".join(parts)


def rating_token_ids(tokenizer, rating_min: int, rating_max: int) -> tuple[list[int], dict[int, list[int]]]:
    ratings = list(range(rating_min, rating_max + 1))
    token_ids: dict[int, list[int]] = {}
    for rating in ratings:
        ids = set()
        for text in (str(rating), " " + str(rating)):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.add(int(encoded[0]))
        token_ids[rating] = sorted(ids)
    all_ids = sorted({token_id for ids in token_ids.values() for token_id in ids})
    if not all_ids:
        raise ValueError("no rating token ids found")
    return all_ids, token_ids


def score_prompts(
    prompts: list[str],
    *,
    model_name: str,
    batch_size: int,
    rating_min: int,
    rating_max: int,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    max_model_len: int | None,
    max_num_seqs: int | None,
    generated_logprobs: int,
    missing_logprob: float,
    temperature: float,
) -> np.ndarray:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_rating_ids, ids_by_rating = rating_token_ids(tokenizer, rating_min, rating_max)
    ratings = list(range(rating_min, rating_max + 1))

    scores = np.full(len(prompts), np.nan, dtype=float)
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=temperature,
        logprobs=generated_logprobs,
    )

    offset = 0
    outputs = []
    for start in range(0, len(prompts), batch_size):
        outputs.extend(llm.generate(prompts[start:start + batch_size], sampling))

    def logprob_value(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, "logprob"):
            return float(value.logprob)
        if isinstance(value, dict) and "logprob" in value:
            return float(value["logprob"])
        return float(value)

    for output in outputs:
        if not output.outputs or not output.outputs[0].logprobs:
            offset += 1
            continue
        first_token_logprobs = output.outputs[0].logprobs[0] or {}
        expanded = {
            int(token_id): logprob_value(value)
            for token_id, value in first_token_logprobs.items()
        }
        probs = {
            rating: float(sum(math.exp(expanded.get(token_id, missing_logprob))
                              for token_id in ids_by_rating[rating]))
            for rating in ratings
        }
        total = sum(probs.values())
        if total > 0:
            expected = sum(rating * probs[rating] for rating in ratings) / total
            scores[offset] = (expected - rating_min) / (rating_max - rating_min)
        offset += 1
    return np.nan_to_num(scores, nan=0.5)


def write_predictions(path: Path, examples: list[dict], scores: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "index": [row["index"] for row in examples],
        "deceptive": scores >= threshold,
        "score": np.clip(scores, 0.0, 1.0),
    }).to_csv(path, index=False)


def mean_metrics(results: list[DatasetResult]) -> dict[str, float | None]:
    out = {}
    for key in METRIC_KEYS:
        values = [result.metrics.get(key) for result in results]
        vals = [float(v) for v in values if v is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def fmt_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-"


@hydra.main(version_base=None, config_path="../../configs", config_name="blackbox_judge")
def main(cfg: DictConfig) -> None:
    original_cwd = Path(get_original_cwd()).resolve()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    split_config = resolve_path(str(cfg.splits_dir), original_cwd) / f"dry.{cfg.split}.yaml"
    if not split_config.exists():
        raise SystemExit(f"{split_config} does not exist. Run scripts/make_dev_splits.py first.")

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(str(cfg.output_dir), original_cwd)
    run_dir = output_dir / str(cfg.method) / str(cfg.split) / run_id
    datasets = load_split_config(split_config, original_cwd)
    few_shot_prefix = build_few_shot_prefix(
        resolve_path(str(cfg.splits_dir), original_cwd),
        split=str(cfg.shots.split),
        n_per_label=int(cfg.shots.n_per_label),
        seed=int(cfg.shots.seed),
        rating_min=int(cfg.judge.rating_min),
        rating_max=int(cfg.judge.rating_max),
        max_prompt_chars=int(cfg.shots.max_prompt_chars),
        base=original_cwd,
    )
    prompt_template = str(cfg.judge.prompt)
    if few_shot_prefix:
        prompt_template = f"{prompt_template}\n\n{few_shot_prefix}"
    results: list[DatasetResult] = []

    for dataset_cfg in datasets:
        labels = load_label_subset(dataset_cfg)
        if cfg.limit is not None:
            labels = labels.iloc[:int(cfg.limit)].copy()
        examples = load_examples_for_labels(dataset_cfg.name, labels, None)
        prompts = [
            build_prompt(
                row["messages"],
                prompt_template,
                int(cfg.judge.max_prompt_chars),
            )
            for row in examples
        ]
        print(f"{dataset_cfg.name}: scoring {len(prompts)} rows with {cfg.judge.model}")
        scores = score_prompts(
            prompts,
            model_name=str(cfg.judge.model),
            batch_size=int(cfg.judge.batch_size),
            rating_min=int(cfg.judge.rating_min),
            rating_max=int(cfg.judge.rating_max),
            dtype=str(cfg.judge.dtype),
            tensor_parallel_size=int(cfg.judge.tensor_parallel_size),
            gpu_memory_utilization=float(cfg.judge.gpu_memory_utilization),
            trust_remote_code=bool(cfg.judge.trust_remote_code),
            max_model_len=None if cfg.judge.max_model_len is None else int(cfg.judge.max_model_len),
            max_num_seqs=None if cfg.judge.max_num_seqs is None else int(cfg.judge.max_num_seqs),
            generated_logprobs=int(cfg.judge.generated_logprobs),
            missing_logprob=float(cfg.judge.missing_logprob),
            temperature=float(cfg.judge.temperature),
        )
        pred_path = run_dir / "predictions" / f"{dataset_cfg.name.replace('/', '__')}.csv"
        write_predictions(pred_path, examples, scores, float(cfg.scoring.threshold))
        preds = load_predictions(pred_path)
        metrics = compute_metrics(preds, labels, partial=True)
        result = DatasetResult(
            dataset=dataset_cfg.name,
            n=len(examples),
            metrics=metrics,
            predictions_path=pred_path.as_posix(),
        )
        results.append(result)
        print(f"  balanced_accuracy={fmt_metric(metrics.get('balanced_accuracy'))} "
              f"auroc={fmt_metric(metrics.get('auroc'))}")

    record = {
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": str(cfg.method),
        "split": str(cfg.split),
        "metrics": mean_metrics(results),
        "config": resolved,
        "datasets": [asdict(result) for result in results],
    }
    config_path = run_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    append_jsonl(resolve_path(str(cfg.ledger), original_cwd), record)

    metrics = record["metrics"]
    print(f"mean balanced_accuracy={fmt_metric(metrics.get('balanced_accuracy'))} "
          f"auroc={fmt_metric(metrics.get('auroc'))}")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
