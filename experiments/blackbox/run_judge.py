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
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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


@dataclass
class DatasetWork:
    config: DatasetConfig
    labels: pd.DataFrame
    examples: list[dict]
    prompts: list[str]
    prompt_count: int


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


def final_assistant_reasoning(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("reasoning") or ""
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


def truncate_context(context: str, max_chars: int, mode: str) -> str:
    if len(context) <= max_chars:
        return context
    if max_chars <= 0:
        return ""
    if mode == "tail":
        return context[-max_chars:]
    if mode == "head":
        return context[:max_chars]
    if mode == "head_tail":
        marker = "\n\n[... middle truncated ...]\n\n"
        available = max_chars - len(marker)
        if available <= 0:
            return context[:max_chars]
        head_chars = available // 2
        tail_chars = available - head_chars
        return context[:head_chars] + marker + context[-tail_chars:]
    raise ValueError(f"unknown judge.context_truncation={mode!r}")


def format_dialogue(
    messages: list[dict],
    max_chars: int,
    truncation: str = "tail",
    context_roles: set[str] | None = None,
) -> tuple[str, str]:
    merged = merge_messages(messages)
    assistant = final_assistant(merged)
    context_messages = [
        m for m in merged
        if context_roles is None or m.get("role", "") in context_roles
    ]
    lines = [f"{m.get('role', '').upper()}: {m.get('content', '')}" for m in context_messages]
    context = "\n\n".join(lines)
    context = truncate_context(context, max_chars, truncation)
    return context, assistant


def build_prompt(
    messages: list[dict],
    prompt_template: str,
    max_chars: int,
    *,
    append_prefix: str | None,
    context_truncation: str = "tail",
    context_roles: set[str] | None = None,
    include_reasoning: bool = False,
    reasoning_max_chars: int = 0,
    reasoning_truncation: str = "tail",
    reference_material: str = "",
) -> str:
    context, output = format_dialogue(messages, max_chars, context_truncation, context_roles)
    prompt = f"{prompt_template}\n\n<context>\n{context}\n</context>\n\n<output>\n{output}\n</output>"
    if include_reasoning and reasoning_max_chars > 0:
        reasoning = final_assistant_reasoning(messages)
        if reasoning:
            reasoning = truncate_context(reasoning, reasoning_max_chars, reasoning_truncation)
            prompt = f"{prompt}\n\n<assistant_reasoning>\n{reasoning}\n</assistant_reasoning>"
    if reference_material:
        prompt = (
            f"{prompt}\n\n<reference_material>\n"
            "The following retrieved text may be incomplete or irrelevant. "
            "Use it only when it clearly matches the question.\n"
            f"{reference_material}\n</reference_material>"
        )
    if append_prefix:
        prompt = f"{prompt}\n\n{append_prefix}"
    return prompt


def load_retrieval_cache(path: Path | None) -> dict[tuple[str, Any], str]:
    """Load cached passages keyed by dataset and row index."""
    if path is None:
        return {}
    references: dict[tuple[str, Any], str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        passages = record.get("passages") or []
        text = "\n".join(
            f"- {passage.get('title', '')}: {passage.get('text', '')}"
            for passage in passages
            if passage.get("text")
        )
        references[(record["dataset"], record["index"])] = text
    return references


def prompt_templates_from_config(cfg: DictConfig, few_shot_prefix: str) -> list[tuple[str, str]]:
    if not bool(OmegaConf.select(cfg, "ensemble.enabled", default=False)):
        prompt_template = str(cfg.judge.prompt)
        if few_shot_prefix:
            prompt_template = f"{prompt_template}\n\n{few_shot_prefix}"
        return [("default", prompt_template)]

    members = OmegaConf.select(cfg, "ensemble.members", default=[])
    if not members:
        raise ValueError("ensemble.enabled=true requires at least one ensemble.members entry")

    templates: list[tuple[str, str]] = []
    for index, member in enumerate(members):
        name = str(member.get("name", f"member_{index}"))
        prompt = str(member.prompt)
        if few_shot_prefix:
            prompt = f"{prompt}\n\n{few_shot_prefix}"
        templates.append((name, prompt))
    return templates


def aggregate_scores(scores: np.ndarray, aggregation: str) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError(f"expected 2D score matrix, got shape={scores.shape}")
    if aggregation == "max":
        return scores.max(axis=1)
    if aggregation in {"mean", "avg"}:
        return scores.mean(axis=1)
    if aggregation == "min":
        return scores.min(axis=1)
    if aggregation == "median":
        return np.median(scores, axis=1)
    raise ValueError(f"unknown ensemble aggregation={aggregation!r}")


def score_matrix_from_raw(raw_scores: np.ndarray, *, n_rows: int, prompt_count: int, order: str) -> np.ndarray:
    if order == "row":
        return raw_scores.reshape(n_rows, prompt_count)
    if order == "member":
        return raw_scores.reshape(prompt_count, n_rows).T
    raise ValueError(f"unknown ensemble order={order!r}")


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


def logit_target_ids(tokenizer, target_specs: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    for spec in target_specs:
        texts = list(spec.get("texts", []))
        if not texts:
            raise ValueError(f"logit target {spec!r} has no texts")
        ids = set()
        for text in texts:
            encoded = tokenizer.encode(str(text), add_special_tokens=False)
            if encoded:
                ids.add(int(encoded[0]))
        if not ids:
            raise ValueError(f"logit target {spec!r} produced no token ids")
        targets.append({
            "name": str(spec.get("name", len(targets))),
            "score": float(spec["score"]),
            "ids": sorted(ids),
        })
    all_ids = sorted({token_id for target in targets for token_id in target["ids"]})
    if not all_ids:
        raise ValueError("no logit target token ids found")
    return all_ids, targets


def requested_logprobs_or_default(generated_logprobs: int | None, n_rating_ids: int) -> int:
    requested_logprobs = generated_logprobs or n_rating_ids
    if requested_logprobs != n_rating_ids:
        raise ValueError(
            "judge.generated_logprobs must be null or equal to the number of "
            f"rating token ids ({n_rating_ids}). vLLM requires this when "
            "logprob_token_ids is set."
        )
    return requested_logprobs


def logprob_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def score_from_rating_probs(
    probs: dict[int, float],
    *,
    rating_min: int,
    rating_max: int,
) -> float:
    total = sum(probs.values())
    if total <= 0:
        return 0.5
    expected = sum(rating * probs[rating] for rating in range(rating_min, rating_max + 1)) / total
    return (expected - rating_min) / (rating_max - rating_min)


def rating_to_score(rating: int, *, rating_min: int, rating_max: int) -> float:
    return (rating - rating_min) / (rating_max - rating_min)


def generate_with_optional_batches(llm: Any, prompts: list[str], sampling: Any, batch_size: int | None) -> list[Any]:
    if batch_size is None or batch_size <= 0:
        return list(llm.generate(prompts, sampling))
    outputs = []
    for start in range(0, len(prompts), batch_size):
        outputs.extend(llm.generate(prompts[start:start + batch_size], sampling))
    return outputs


def render_chat_prompts(
    tokenizer: Any,
    prompts: list[str],
    *,
    enable_thinking: bool | None,
) -> list[str]:
    rendered = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            rendered.append(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            rendered.append(tokenizer.apply_chat_template(messages, **kwargs))
    return rendered


def rating_probs_from_logprobs(
    first_token_logprobs: dict[Any, Any],
    *,
    ids_by_rating: dict[int, list[int]],
    ratings: list[int],
    missing_logprob: float,
) -> dict[int, float]:
    expanded = {
        int(token_id): logprob_value(value)
        for token_id, value in first_token_logprobs.items()
    }
    return {
        rating: float(sum(math.exp(expanded.get(token_id, missing_logprob))
                          for token_id in ids_by_rating[rating]))
        for rating in ratings
    }


def target_probs_from_logprobs(
    first_token_logprobs: dict[Any, Any],
    *,
    targets: list[dict[str, Any]],
    missing_logprob: float,
) -> dict[str, float]:
    expanded = {
        int(token_id): logprob_value(value)
        for token_id, value in first_token_logprobs.items()
    }
    return {
        target["name"]: float(sum(math.exp(expanded.get(token_id, missing_logprob))
                                  for token_id in target["ids"]))
        for target in targets
    }


def score_from_target_probs(probs: dict[str, float], targets: list[dict[str, Any]]) -> float:
    total = sum(probs.values())
    if total <= 0:
        return 0.5
    expected = sum(float(target["score"]) * probs[target["name"]] for target in targets) / total
    return expected


def structured_rating_prompt(prompt: str, reasoning: str, rating_instruction: str) -> str:
    return (
        f"{prompt}\n\n"
        "<reasoning>\n"
        f"{reasoning.strip()}\n"
        "</reasoning>\n\n"
        f"{rating_instruction.strip()}\n\n"
        "Rating:"
    )


def vllm_kwargs_from_config(
    *,
    model_name: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    max_model_len: int | None,
    max_num_seqs: int | None,
    spec_method: str | None,
    spec_model: str | None,
    spec_tokens: int | None,
) -> dict[str, Any]:
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    optional = {
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "spec_method": spec_method,
        "spec_model": spec_model,
        "spec_tokens": spec_tokens,
    }
    llm_kwargs.update({key: value for key, value in optional.items() if value is not None})
    return llm_kwargs


def parse_rating(text: str, *, rating_min: int, rating_max: int) -> int | None:
    rating_range = f"[{rating_min}-{rating_max}]"
    patterns = (
        # GPT-OSS Harmony completions can concatenate channel markers with the
        # final response, yielding e.g. ``assistantfinalRating: 7``.
        rf"(?im)assistantfinal\s*rating\s*[:\-]?\s*\**\s*({rating_range})\b",
        rf"(?im)\b(?:final\s+answer|final\s+rating|rating|score)\s*[:\-]?\s*\**\s*({rating_range})\b",
        rf"(?im)\b({rating_range})\s*/\s*{rating_max}\b",
    )
    matches: list[int] = []
    for pattern in patterns:
        matches.extend(int(match) for match in re.findall(pattern, text))
    if matches:
        return matches[-1]
    stripped = text.strip()
    if re.fullmatch(rating_range, stripped):
        return int(stripped)
    return None


class OfflineVllmRatingJudge:
    def __init__(
        self,
        *,
        model_name: str,
        rating_min: int,
        rating_max: int,
        logit_targets: list[dict[str, Any]] | None,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
        max_model_len: int | None,
        max_num_seqs: int | None,
        spec_method: str | None,
        spec_model: str | None,
        spec_tokens: int | None,
        generated_logprobs: int | None,
        missing_logprob: float,
        temperature: float,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.rating_min = rating_min
        self.rating_max = rating_max
        self.missing_logprob = missing_logprob
        self.ratings = list(range(rating_min, rating_max + 1))
        self.targets: list[dict[str, Any]] | None = None

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if logit_targets:
            self.all_rating_ids, self.targets = logit_target_ids(tokenizer, logit_targets)
            self.ids_by_rating = {}
        else:
            self.all_rating_ids, self.ids_by_rating = rating_token_ids(tokenizer, rating_min, rating_max)
        requested_logprobs = requested_logprobs_or_default(generated_logprobs, len(self.all_rating_ids))

        llm_kwargs = vllm_kwargs_from_config(
            model_name=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            spec_method=spec_method,
            spec_model=spec_model,
            spec_tokens=spec_tokens,
        )
        self.llm = LLM(**llm_kwargs)
        self.sampling = SamplingParams(
            max_tokens=1,
            temperature=temperature,
            logprobs=requested_logprobs,
            logprob_token_ids=self.all_rating_ids,
            allowed_token_ids=self.all_rating_ids,
        )

    def score_prompts(self, prompts: list[str], *, batch_size: int | None) -> np.ndarray:
        scores = np.full(len(prompts), np.nan, dtype=float)
        outputs = generate_with_optional_batches(self.llm, prompts, self.sampling, batch_size)

        for offset, output in enumerate(outputs):
            if not output.outputs or not output.outputs[0].logprobs:
                continue
            first_token_logprobs = output.outputs[0].logprobs[0] or {}
            if self.targets is not None:
                target_probs = target_probs_from_logprobs(
                    first_token_logprobs,
                    targets=self.targets,
                    missing_logprob=self.missing_logprob,
                )
                scores[offset] = score_from_target_probs(target_probs, self.targets)
            else:
                probs = rating_probs_from_logprobs(
                    first_token_logprobs,
                    ids_by_rating=self.ids_by_rating,
                    ratings=self.ratings,
                    missing_logprob=self.missing_logprob,
                )
                scores[offset] = score_from_rating_probs(
                    probs,
                    rating_min=self.rating_min,
                    rating_max=self.rating_max,
                )
        return np.nan_to_num(scores, nan=0.5)


class OfflineVllmGenerateJudge:
    def __init__(
        self,
        *,
        model_name: str,
        rating_min: int,
        rating_max: int,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
        max_model_len: int | None,
        max_num_seqs: int | None,
        spec_method: str | None,
        spec_model: str | None,
        spec_tokens: int | None,
        max_tokens: int,
        temperature: float,
        use_chat_template: bool,
        enable_thinking: bool | None,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.rating_min = rating_min
        self.rating_max = rating_max
        self.generations: list[dict[str, Any]] = []
        self.parse_error_count = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) if use_chat_template else None
        self.enable_thinking = enable_thinking
        llm_kwargs = vllm_kwargs_from_config(
            model_name=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            spec_method=spec_method,
            spec_model=spec_model,
            spec_tokens=spec_tokens,
        )
        self.llm = LLM(**llm_kwargs)
        self.sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def score_prompts(self, prompts: list[str], *, batch_size: int | None) -> np.ndarray:
        generation_prompts = prompts
        if self.tokenizer is not None:
            generation_prompts = render_chat_prompts(
                self.tokenizer,
                prompts,
                enable_thinking=self.enable_thinking,
            )
        outputs = generate_with_optional_batches(self.llm, generation_prompts, self.sampling, batch_size)

        scores = np.zeros(len(prompts), dtype=float)
        self.generations = []
        self.parse_error_count = 0
        for offset, output in enumerate(outputs):
            text = output.outputs[0].text if output.outputs else ""
            rating = parse_rating(text, rating_min=self.rating_min, rating_max=self.rating_max)
            parse_error = rating is None
            if parse_error:
                self.parse_error_count += 1
            else:
                scores[offset] = rating_to_score(
                    rating,
                    rating_min=self.rating_min,
                    rating_max=self.rating_max,
                )
            self.generations.append({
                "offset": offset,
                "rating": rating,
                "parse_error": parse_error,
                "text": text,
            })
        return scores


class OfflineVllmStructuredJudge:
    def __init__(
        self,
        *,
        model_name: str,
        rating_min: int,
        rating_max: int,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
        max_model_len: int | None,
        max_num_seqs: int | None,
        spec_method: str | None,
        spec_model: str | None,
        spec_tokens: int | None,
        generated_logprobs: int | None,
        missing_logprob: float,
        max_tokens: int,
        temperature: float,
        final_rating_prompt: str,
        use_chat_template: bool,
        enable_thinking: bool | None,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.rating_min = rating_min
        self.rating_max = rating_max
        self.missing_logprob = missing_logprob
        self.ratings = list(range(rating_min, rating_max + 1))
        self.final_rating_prompt = final_rating_prompt
        self.generations: list[dict[str, Any]] = []
        self.parse_error_count = 0

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer if use_chat_template else None
        self.enable_thinking = enable_thinking
        self.all_rating_ids, self.ids_by_rating = rating_token_ids(tokenizer, rating_min, rating_max)
        requested_logprobs = requested_logprobs_or_default(generated_logprobs, len(self.all_rating_ids))

        llm_kwargs = vllm_kwargs_from_config(
            model_name=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            spec_method=spec_method,
            spec_model=spec_model,
            spec_tokens=spec_tokens,
        )
        self.llm = LLM(**llm_kwargs)
        self.reasoning_sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.rating_sampling = SamplingParams(
            max_tokens=1,
            temperature=temperature,
            logprobs=requested_logprobs,
            logprob_token_ids=self.all_rating_ids,
            allowed_token_ids=self.all_rating_ids,
        )

    def score_prompts(self, prompts: list[str], *, batch_size: int | None) -> np.ndarray:
        reasoning_prompts = prompts
        if self.tokenizer is not None:
            reasoning_prompts = render_chat_prompts(
                self.tokenizer,
                prompts,
                enable_thinking=self.enable_thinking,
            )
        reasoning_outputs = generate_with_optional_batches(
            self.llm,
            reasoning_prompts,
            self.reasoning_sampling,
            batch_size,
        )
        reasoning_texts = [
            output.outputs[0].text if output.outputs else ""
            for output in reasoning_outputs
        ]
        rating_prompts = [
            structured_rating_prompt(prompt, reasoning, self.final_rating_prompt)
            for prompt, reasoning in zip(prompts, reasoning_texts, strict=True)
        ]
        if self.tokenizer is not None:
            rating_prompts = render_chat_prompts(
                self.tokenizer,
                rating_prompts,
                enable_thinking=self.enable_thinking,
            )
        rating_outputs = generate_with_optional_batches(
            self.llm,
            rating_prompts,
            self.rating_sampling,
            batch_size,
        )

        scores = np.full(len(prompts), np.nan, dtype=float)
        self.generations = []
        self.parse_error_count = 0
        for offset, (reasoning, output) in enumerate(zip(reasoning_texts, rating_outputs, strict=True)):
            first_token_logprobs = {}
            if output.outputs and output.outputs[0].logprobs:
                first_token_logprobs = output.outputs[0].logprobs[0] or {}
            probs = rating_probs_from_logprobs(
                first_token_logprobs,
                ids_by_rating=self.ids_by_rating,
                ratings=self.ratings,
                missing_logprob=self.missing_logprob,
            )
            score = score_from_rating_probs(
                probs,
                rating_min=self.rating_min,
                rating_max=self.rating_max,
            )
            best_rating = max(probs, key=probs.get) if probs else None
            scores[offset] = score
            self.generations.append({
                "offset": offset,
                "rating": best_rating,
                "parse_error": False,
                "score": score,
                "text": reasoning,
            })
        return np.nan_to_num(scores, nan=0.5)


class OpenAIRatingJudge:
    def __init__(
        self,
        *,
        model_name: str,
        served_model: str,
        api_base: str,
        api_key: str,
        concurrency: int,
        request_timeout: float,
        rating_min: int,
        rating_max: int,
        generated_logprobs: int | None,
        temperature: float,
    ) -> None:
        from transformers import AutoTokenizer

        self.served_model = served_model
        self.endpoint = api_base.rstrip("/") + "/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.concurrency = concurrency
        self.request_timeout = request_timeout
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.ratings = list(range(rating_min, rating_max + 1))
        self.temperature = temperature

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.all_rating_ids, self.ids_by_rating = rating_token_ids(tokenizer, rating_min, rating_max)
        self.rating_by_token_id = {
            token_id: rating
            for rating, ids in self.ids_by_rating.items()
            for token_id in ids
        }
        self.requested_logprobs = requested_logprobs_or_default(
            generated_logprobs,
            len(self.all_rating_ids),
        )

    def score_prompts(self, prompts: list[str], *, batch_size: int | None) -> np.ndarray:
        del batch_size
        workers = max(1, self.concurrency)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scores = list(executor.map(self._score_prompt, prompts))
        return np.array(scores, dtype=float)

    def _score_prompt(self, prompt: str) -> float:
        import requests

        payload = {
            "model": self.served_model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": self.temperature,
            "logprobs": self.requested_logprobs,
            "allowed_token_ids": self.all_rating_ids,
            "return_tokens_as_token_ids": True,
        }
        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        top_logprobs = data["choices"][0].get("logprobs", {}).get("top_logprobs", [{}])[0] or {}
        probs = {rating: 0.0 for rating in self.ratings}
        for token, value in top_logprobs.items():
            rating = self._rating_from_response_token(token)
            if rating is not None:
                probs[rating] += math.exp(logprob_value(value))
        return score_from_rating_probs(
            probs,
            rating_min=self.rating_min,
            rating_max=self.rating_max,
        )

    def _rating_from_response_token(self, token: str) -> int | None:
        if token.startswith("token_id:"):
            token_id = int(token.removeprefix("token_id:"))
            return self.rating_by_token_id.get(token_id)
        stripped = token.strip()
        if stripped.isdigit():
            rating = int(stripped)
            if self.rating_min <= rating <= self.rating_max:
                return rating
        return None


class OpenAIGenerateJudge:
    def __init__(
        self,
        *,
        model_name: str,
        served_model: str,
        api_base: str,
        api_key: str,
        concurrency: int,
        request_timeout: float,
        rating_min: int,
        rating_max: int,
        max_tokens: int,
        temperature: float,
        use_chat_template: bool,
        enable_thinking: bool | None,
    ) -> None:
        self.served_model = served_model
        self.endpoint = api_base.rstrip("/") + "/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.concurrency = concurrency
        self.request_timeout = request_timeout
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer = None
        if use_chat_template:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.enable_thinking = enable_thinking
        self.generations: list[dict[str, Any]] = []
        self.parse_error_count = 0

    def score_prompts(self, prompts: list[str], *, batch_size: int | None) -> np.ndarray:
        del batch_size
        generation_prompts = prompts
        if self.tokenizer is not None:
            generation_prompts = render_chat_prompts(
                self.tokenizer,
                prompts,
                enable_thinking=self.enable_thinking,
            )
        workers = max(1, self.concurrency)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            generations = list(executor.map(self._generate_prompt, generation_prompts))

        scores = np.zeros(len(prompts), dtype=float)
        self.generations = []
        self.parse_error_count = 0
        for offset, text in enumerate(generations):
            rating = parse_rating(text, rating_min=self.rating_min, rating_max=self.rating_max)
            parse_error = rating is None
            if parse_error:
                self.parse_error_count += 1
            else:
                scores[offset] = rating_to_score(
                    rating,
                    rating_min=self.rating_min,
                    rating_max=self.rating_max,
                )
            self.generations.append({
                "offset": offset,
                "rating": rating,
                "parse_error": parse_error,
                "text": text,
            })
        return scores

    def _generate_prompt(self, prompt: str) -> str:
        import requests

        payload = {
            "model": self.served_model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0].get("text", ""))


def build_judge(cfg: DictConfig):
    generated_logprobs = (
        None
        if cfg.judge.generated_logprobs is None
        else int(cfg.judge.generated_logprobs)
    )
    backend = str(cfg.judge.backend)
    mode = str(cfg.judge.mode)
    if backend == "offline":
        selected_logit_targets = OmegaConf.select(cfg, "judge.logit_targets", default=None)
        if selected_logit_targets is None:
            logit_targets = []
        elif isinstance(selected_logit_targets, list):
            logit_targets = selected_logit_targets
        else:
            logit_targets = OmegaConf.to_container(selected_logit_targets, resolve=True)
        common = {
            "model_name": str(cfg.judge.model),
            "rating_min": int(cfg.judge.rating_min),
            "rating_max": int(cfg.judge.rating_max),
            "dtype": str(cfg.judge.dtype),
            "tensor_parallel_size": int(cfg.judge.tensor_parallel_size),
            "gpu_memory_utilization": float(cfg.judge.gpu_memory_utilization),
            "trust_remote_code": bool(cfg.judge.trust_remote_code),
            "max_model_len": None if cfg.judge.max_model_len is None else int(cfg.judge.max_model_len),
            "max_num_seqs": None if cfg.judge.max_num_seqs is None else int(cfg.judge.max_num_seqs),
            "spec_method": None if cfg.judge.spec_method is None else str(cfg.judge.spec_method),
            "spec_model": None if cfg.judge.spec_model is None else str(cfg.judge.spec_model),
            "spec_tokens": None if cfg.judge.spec_tokens is None else int(cfg.judge.spec_tokens),
            "temperature": float(cfg.judge.temperature),
        }
        if mode == "logits":
            return OfflineVllmRatingJudge(
                **common,
                logit_targets=logit_targets,
                generated_logprobs=generated_logprobs,
                missing_logprob=float(cfg.judge.missing_logprob),
            )
        if mode == "generate":
            return OfflineVllmGenerateJudge(
                **common,
                max_tokens=int(cfg.judge.max_tokens),
                use_chat_template=bool(OmegaConf.select(cfg, "judge.use_chat_template", default=False)),
                enable_thinking=OmegaConf.select(cfg, "judge.enable_thinking", default=None),
            )
        if mode == "structured":
            return OfflineVllmStructuredJudge(
                **common,
                generated_logprobs=generated_logprobs,
                missing_logprob=float(cfg.judge.missing_logprob),
                max_tokens=int(cfg.judge.max_tokens),
                final_rating_prompt=str(cfg.judge.structured_rating_prompt),
                use_chat_template=bool(OmegaConf.select(cfg, "judge.use_chat_template", default=False)),
                enable_thinking=OmegaConf.select(cfg, "judge.enable_thinking", default=None),
            )
        raise ValueError(f"unknown judge.mode={mode!r}")
    if backend == "openai":
        if mode == "logits":
            return OpenAIRatingJudge(
                model_name=str(cfg.judge.model),
                served_model=str(cfg.judge.served_model),
                api_base=str(cfg.judge.api_base),
                api_key=str(cfg.judge.api_key),
                concurrency=int(cfg.judge.api_concurrency),
                request_timeout=float(cfg.judge.request_timeout),
                rating_min=int(cfg.judge.rating_min),
                rating_max=int(cfg.judge.rating_max),
                generated_logprobs=generated_logprobs,
                temperature=float(cfg.judge.temperature),
            )
        if mode == "generate":
            return OpenAIGenerateJudge(
                model_name=str(cfg.judge.model),
                served_model=str(cfg.judge.served_model),
                api_base=str(cfg.judge.api_base),
                api_key=str(cfg.judge.api_key),
                concurrency=int(cfg.judge.api_concurrency),
                request_timeout=float(cfg.judge.request_timeout),
                rating_min=int(cfg.judge.rating_min),
                rating_max=int(cfg.judge.rating_max),
                max_tokens=int(cfg.judge.max_tokens),
                temperature=float(cfg.judge.temperature),
                use_chat_template=bool(OmegaConf.select(cfg, "judge.use_chat_template", default=False)),
                enable_thinking=OmegaConf.select(cfg, "judge.enable_thinking", default=None),
            )
        raise ValueError(f"judge.backend=openai does not support judge.mode={mode!r}")
    raise ValueError(f"unknown judge.backend={backend!r}")


def write_predictions(path: Path, examples: list[dict], scores: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "index": [row["index"] for row in examples],
        "deceptive": scores >= threshold,
        "score": np.clip(scores, 0.0, 1.0),
    }).to_csv(path, index=False)


def write_member_predictions(
    run_dir: Path,
    dataset_name: str,
    examples: list[dict],
    score_matrix: np.ndarray,
    member_names: list[str],
    threshold: float,
) -> dict[str, str]:
    """Persist per-member scores for ensemble runs."""
    paths: dict[str, str] = {}
    dataset_file = f"{dataset_name.replace('/', '__')}.csv"
    for member_index, member_name in enumerate(member_names):
        safe_member = "".join(
            c if c.isalnum() or c in "_.=-" else "_"
            for c in member_name
        )
        path = run_dir / "member_predictions" / safe_member / dataset_file
        write_predictions(path, examples, score_matrix[:, member_index], threshold)
        paths[member_name] = path.as_posix()
    return paths


def write_generations(path: Path, metadata: list[dict[str, Any]], generations: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for meta, generation in zip(metadata, generations, strict=True):
            f.write(json.dumps({**meta, **generation}, sort_keys=True) + "\n")


def json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def mean_metrics(results: list[DatasetResult]) -> dict[str, float | None]:
    out = {}
    for key in METRIC_KEYS:
        values = [result.metrics.get(key) for result in results]
        vals = [float(v) for v in values if v is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def fmt_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-"


def fmt_seconds(value: float | None) -> str:
    return f"{value:.1f}s" if isinstance(value, (int, float)) else "-"


def fmt_rate(value: float | None) -> str:
    return f"{value:.1f}/s" if isinstance(value, (int, float)) else "-"


def fmt_submitted_at(value: object) -> str:
    if not isinstance(value, str):
        return "-"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return value.split(".")[0].replace("T", " ")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed.isoformat(sep=" ", timespec="seconds")


def render_leaderboard(results_root: Path, output_path: Path) -> None:
    records = [
        json.loads(path.read_text())
        for path in results_root.glob("*/*/result.json")
    ]
    records = [record for record in records if record.get("split") == "test"]
    records.sort(key=lambda row: str(row.get("submitted_at", "")), reverse=True)

    lines = [
        "# Black-Box Experiment Leaderboard",
        "",
        "Timing is scoring-only wall time: it excludes vLLM startup/model load/compile and dataset preparation.",
        "Submitted timestamps are UTC.",
        "",
        "| submitted_at | method | AUROC | bal_acc | recall | FPR | score_time | rows/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record.get("metrics", {})
        timing = record.get("timing", {})
        lines.append(
            "| "
            f"{fmt_submitted_at(record.get('submitted_at'))} | "
            f"{record.get('method', '-')} | "
            f"{fmt_metric(metrics.get('auroc'))} | "
            f"{fmt_metric(metrics.get('balanced_accuracy'))} | "
            f"{fmt_metric(metrics.get('recall'))} | "
            f"{fmt_metric(metrics.get('fpr'))} | "
            f"{fmt_seconds(timing.get('score_seconds'))} | "
            f"{fmt_rate(timing.get('rows_per_second'))} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


@hydra.main(version_base=None, config_path="../../configs", config_name="blackbox_judge")
def main(cfg: DictConfig) -> None:
    original_cwd = Path(get_original_cwd()).resolve()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    split_config = resolve_path(str(cfg.splits_dir), original_cwd) / f"dry.{cfg.split}.yaml"
    if not split_config.exists():
        raise SystemExit(f"{split_config} does not exist. Run scripts/make_dev_splits.py first.")

    output_dir = resolve_path(str(cfg.output_dir), original_cwd)
    run_dir = output_dir / str(cfg.method) / str(cfg.split)
    retrieval_cache = OmegaConf.select(cfg, "retrieval_cache", default=None)
    references = load_retrieval_cache(
        None if retrieval_cache is None else resolve_path(str(retrieval_cache), original_cwd)
    )
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
    prompt_templates = prompt_templates_from_config(cfg, few_shot_prefix)
    prompt_count = len(prompt_templates)
    ensemble_enabled = prompt_count > 1
    aggregation = str(OmegaConf.select(cfg, "ensemble.aggregation", default="max"))
    ensemble_order = str(OmegaConf.select(cfg, "ensemble.order", default="row"))
    if ensemble_order not in {"row", "member"}:
        raise ValueError(f"unknown ensemble.order={ensemble_order!r}")
    append_prefix = None
    if str(cfg.judge.mode) == "logits":
        append_prefix = str(OmegaConf.select(cfg, "judge.logit_prefix", default="Rating:"))
    context_truncation = str(OmegaConf.select(cfg, "judge.context_truncation", default="tail"))
    selected_context_roles = OmegaConf.select(cfg, "judge.context_roles", default=None)
    context_roles = None
    if selected_context_roles is not None:
        context_roles = set(str(role) for role in selected_context_roles)
    include_reasoning = bool(OmegaConf.select(cfg, "judge.include_reasoning", default=False))
    reasoning_max_chars = int(OmegaConf.select(cfg, "judge.reasoning_max_chars", default=0))
    reasoning_truncation = str(OmegaConf.select(cfg, "judge.reasoning_truncation", default="tail"))
    results: list[DatasetResult] = []
    print(f"initializing {cfg.judge.backend} judge for {cfg.judge.model}")
    judge = build_judge(cfg)
    dataset_work: list[DatasetWork] = []
    all_prompts: list[str] = []
    all_metadata: list[dict[str, Any]] = []

    for dataset_cfg in datasets:
        labels = load_label_subset(dataset_cfg)
        if cfg.limit is not None:
            labels = labels.iloc[:int(cfg.limit)].copy()
        examples = load_examples_for_labels(dataset_cfg.name, labels, None)
        prompts = []
        label_by_index = {row["index"]: int(row["label"]) for row in labels.to_dict("records")}
        prompt_items: list[tuple[dict, int, str, str]]
        if ensemble_order == "member":
            prompt_items = [
                (row, member_index, member_name, prompt_template)
                for member_index, (member_name, prompt_template) in enumerate(prompt_templates)
                for row in examples
            ]
        else:
            prompt_items = [
                (row, member_index, member_name, prompt_template)
                for row in examples
                for member_index, (member_name, prompt_template) in enumerate(prompt_templates)
            ]
        for row, member_index, member_name, prompt_template in prompt_items:
            prompts.append(
                build_prompt(
                    row["messages"],
                    prompt_template,
                    int(cfg.judge.max_prompt_chars),
                    append_prefix=append_prefix,
                    context_truncation=context_truncation,
                    context_roles=context_roles,
                    include_reasoning=include_reasoning,
                    reasoning_max_chars=reasoning_max_chars,
                    reasoning_truncation=reasoning_truncation,
                    reference_material=references.get((dataset_cfg.name, row["index"]), ""),
                )
            )
            metadata = {
                "dataset": dataset_cfg.name,
                "index": json_scalar(row["index"]),
                "label": label_by_index[row["index"]],
            }
            if ensemble_enabled:
                metadata["ensemble_member"] = member_name
                metadata["ensemble_member_index"] = member_index
            all_metadata.append(metadata)
        dataset_work.append(DatasetWork(dataset_cfg, labels, examples, prompts, prompt_count))
        all_prompts.extend(prompts)
        print(f"{dataset_cfg.name}: prepared {len(examples)} rows"
              f" ({len(prompts)} prompt evaluations)")

    batch_size = None if cfg.judge.batch_size is None else int(cfg.judge.batch_size)
    n_rows = sum(len(work.examples) for work in dataset_work)
    print(f"scoring {len(all_prompts)} prompt evaluations for {n_rows} rows "
          f"across {len(dataset_work)} datasets with {cfg.judge.model}")
    score_start = time.perf_counter()
    all_scores = judge.score_prompts(all_prompts, batch_size=batch_size)
    score_seconds = time.perf_counter() - score_start

    offset = 0
    member_names = [name for name, _ in prompt_templates]
    member_predictions: dict[str, dict[str, str]] = {}
    for work in dataset_work:
        end = offset + len(work.prompts)
        raw_scores = all_scores[offset:end]
        offset = end
        if work.prompt_count > 1:
            score_matrix = score_matrix_from_raw(
                raw_scores,
                n_rows=len(work.examples),
                prompt_count=work.prompt_count,
                order=ensemble_order,
            )
            member_predictions[work.config.name] = write_member_predictions(
                run_dir,
                work.config.name,
                work.examples,
                score_matrix,
                member_names,
                float(cfg.scoring.threshold),
            )
            scores = aggregate_scores(score_matrix, aggregation)
        else:
            scores = raw_scores
        pred_path = run_dir / "predictions" / f"{work.config.name.replace('/', '__')}.csv"
        write_predictions(pred_path, work.examples, scores, float(cfg.scoring.threshold))
        preds = load_predictions(pred_path)
        metrics = compute_metrics(preds, work.labels, partial=True)
        result = DatasetResult(
            dataset=work.config.name,
            n=len(work.examples),
            metrics=metrics,
            predictions_path=pred_path.as_posix(),
        )
        results.append(result)
        print(f"  balanced_accuracy={fmt_metric(metrics.get('balanced_accuracy'))} "
              f"auroc={fmt_metric(metrics.get('auroc'))}")

    result_path = run_dir / "result.json"
    rows_per_second = n_rows / score_seconds if score_seconds > 0 else None
    prompt_evaluations_per_second = len(all_prompts) / score_seconds if score_seconds > 0 else None
    generation_path = None
    generations = getattr(judge, "generations", None)
    if generations is not None:
        generation_path = run_dir / "generations.jsonl"
        write_generations(generation_path, all_metadata, generations)
    record = {
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": str(cfg.method),
        "split": str(cfg.split),
        "n": n_rows,
        "n_prompt_evaluations": len(all_prompts),
        "metrics": mean_metrics(results),
        "timing": {
            "score_seconds": score_seconds,
            "rows_per_second": rows_per_second,
            "prompt_evaluations_per_second": prompt_evaluations_per_second,
            "note": "excludes judge initialization/model startup and dataset preparation",
        },
        "config": resolved,
        "datasets": [asdict(result) for result in results],
        "run_dir": run_dir.as_posix(),
        "result_path": result_path.as_posix(),
        "parse_errors": getattr(judge, "parse_error_count", 0),
        "generations_path": None if generation_path is None else generation_path.as_posix(),
    }
    if ensemble_enabled:
        record["ensemble"] = {
            "aggregation": aggregation,
            "order": ensemble_order,
            "members": member_names,
        }
        record["member_predictions"] = member_predictions
    config_path = run_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    render_leaderboard(output_dir, resolve_path(str(cfg.leaderboard), original_cwd))

    metrics = record["metrics"]
    print(f"mean balanced_accuracy={fmt_metric(metrics.get('balanced_accuracy'))} "
          f"auroc={fmt_metric(metrics.get('auroc'))}")
    print(f"score_time={fmt_seconds(score_seconds)} rows_per_second={fmt_rate(rows_per_second)}")
    if ensemble_enabled:
        print(f"prompt_evaluations_per_second={fmt_rate(prompt_evaluations_per_second)}")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
