#!/usr/bin/env python3
"""Evaluate a saved Qwen GRPO LoRA adapter on a local labeled split."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
HF_SPLIT = "test"


@dataclasses.dataclass
class SplitRecords:
    frame: pd.DataFrame
    dataset_names: list[str]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    name: str
    labels_uri: str
    id_column: str = "index"
    label_column: str = "deceptive"


@dataclasses.dataclass
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


def load_labels(cfg: DatasetConfig) -> pd.DataFrame:
    uri = Path(cfg.labels_uri)
    if cfg.labels_uri.endswith(".csv") and uri.exists():
        raw = pd.read_csv(uri)
    else:
        from datasets import load_dataset

        raw = load_dataset(cfg.labels_uri, split=HF_SPLIT).to_pandas()
    if cfg.id_column not in raw.columns:
        raw = raw.reset_index().rename(columns={"index": cfg.id_column})
    return pd.DataFrame({
        "index": raw[cfg.id_column],
        "label": raw[cfg.label_column].astype(int),
    })


def merge_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def final_assistant(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def final_assistant_reasoning(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("reasoning") or "")
    return ""


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
    raise ValueError(f"unknown truncation mode={mode!r}")


def format_dialogue(
    messages: list[dict[str, Any]],
    max_chars: int,
    context_truncation: str,
) -> tuple[str, str]:
    merged = merge_messages(messages)
    assistant = final_assistant(merged)
    context = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in merged
    )
    return truncate_context(context, max_chars, context_truncation), assistant


def build_prompt(
    messages: list[dict[str, Any]],
    prompt_template: str,
    max_chars: int,
    *,
    context_truncation: str,
    include_reasoning: bool,
    reasoning_max_chars: int,
    reasoning_truncation: str,
) -> str:
    context, output = format_dialogue(messages, max_chars, context_truncation)
    prompt = f"{prompt_template}\n\n<context>\n{context}\n</context>\n\n<output>\n{output}\n</output>"
    if include_reasoning and reasoning_max_chars > 0:
        reasoning = truncate_context(
            final_assistant_reasoning(messages),
            reasoning_max_chars,
            reasoning_truncation,
        )
        if reasoning:
            prompt = f"{prompt}\n\n<assistant_reasoning>\n{reasoning}\n</assistant_reasoning>"
    return prompt


def render_chat_prompt(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_examples_for_labels(
    dataset_name: str,
    labels: pd.DataFrame,
    *,
    prompt_template: str,
    tokenizer: Any,
    max_prompt_chars: int,
    context_truncation: str,
    include_reasoning: bool,
    reasoning_max_chars: int,
    reasoning_truncation: str,
    enable_thinking: bool,
) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=HF_SPLIT)
    if "index" not in ds.column_names:
        ds = ds.add_column("index", list(range(len(ds))))

    wanted = set(labels["index"].tolist())
    label_by_index = dict(zip(labels["index"], labels["label"], strict=True))
    rows = []
    for row in ds:
        index = row["index"]
        if index not in wanted:
            continue
        raw_prompt = build_prompt(
            row["messages"],
            prompt_template,
            max_prompt_chars,
            context_truncation=context_truncation,
            include_reasoning=include_reasoning,
            reasoning_max_chars=reasoning_max_chars,
            reasoning_truncation=reasoning_truncation,
        )
        rows.append({
            "dataset": dataset_name,
            "index": index,
            "label": int(label_by_index[index]),
            "prompt": render_chat_prompt(
                tokenizer,
                raw_prompt,
                enable_thinking=enable_thinking,
            ),
        })

    order = {idx: i for i, idx in enumerate(labels["index"].tolist())}
    rows.sort(key=lambda item: order[item["index"]])
    if len(rows) != len(labels):
        raise RuntimeError(
            f"{dataset_name}: loaded {len(rows)} examples for {len(labels)} labels"
        )
    return pd.DataFrame(rows)


def load_split(
    split: str,
    splits_dir: Path,
    *,
    prompt_template: str,
    tokenizer: Any,
    max_prompt_chars: int,
    context_truncation: str,
    include_reasoning: bool,
    reasoning_max_chars: int,
    reasoning_truncation: str,
    enable_thinking: bool,
) -> SplitRecords:
    configs = load_split_config(splits_dir / f"dry.{split}.yaml", ROOT)
    frames = []
    for cfg in configs:
        labels = load_labels(cfg)
        frames.append(
            load_examples_for_labels(
                cfg.name,
                labels,
                prompt_template=prompt_template,
                tokenizer=tokenizer,
                max_prompt_chars=max_prompt_chars,
                context_truncation=context_truncation,
                include_reasoning=include_reasoning,
                reasoning_max_chars=reasoning_max_chars,
                reasoning_truncation=reasoning_truncation,
                enable_thinking=enable_thinking,
            )
        )
    return SplitRecords(
        frame=pd.concat(frames, ignore_index=True),
        dataset_names=[cfg.name for cfg in configs],
    )


def apply_global_limit(records: SplitRecords, limit: int | None) -> SplitRecords:
    if limit is None:
        return records
    return SplitRecords(
        frame=records.frame.iloc[:limit].reset_index(drop=True),
        dataset_names=records.dataset_names,
    )


STRUCTURED_COMPLETION_RE = re.compile(
    r"(?is)^\s*<reasoning>\s*(?P<reasoning>.+?)\s*</reasoning>\s*Rating:\s*(?P<rating>[1-7])\s*$"
)


def structured_completion_match(text: str) -> re.Match[str] | None:
    match = STRUCTURED_COMPLETION_RE.fullmatch(text)
    if match is None or not match.group("reasoning").strip():
        return None
    return match


def parse_rating(text: str, *, rating_min: int, rating_max: int) -> int | None:
    structured_match = structured_completion_match(text)
    if structured_match is not None:
        return int(structured_match.group("rating"))

    rating_range = f"[{rating_min}-{rating_max}]"
    patterns = (
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


def rating_to_score(rating: int, *, rating_min: int, rating_max: int) -> float:
    return (rating - rating_min) / (rating_max - rating_min)


def evaluate_model(
    *,
    model: Any,
    tokenizer: Any,
    records: SplitRecords,
    batch_size: int,
    max_new_tokens: int,
    rating_min: int,
    rating_max: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    device = model.device
    rows = []
    parse_errors = 0
    started = time.time()
    model.eval()

    import torch

    with torch.inference_mode():
        for start in range(0, len(records.frame), batch_size):
            batch = records.frame.iloc[start:start + batch_size]
            encoded = tokenizer(
                batch["prompt"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            prompt_len = encoded["input_ids"].shape[1]
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = outputs[:, prompt_len:]
            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            for (_, row), text in zip(batch.iterrows(), texts, strict=True):
                rating = parse_rating(text, rating_min=rating_min, rating_max=rating_max)
                if rating is None:
                    parse_errors += 1
                    score = 0.0
                else:
                    score = rating_to_score(rating, rating_min=rating_min, rating_max=rating_max)
                rows.append({
                    "dataset": row["dataset"],
                    "index": row["index"],
                    "label": int(row["label"]),
                    "score": float(score),
                    "rating": rating,
                    "parse_error": rating is None,
                    "format_valid": structured_completion_match(text) is not None,
                    "generation": text,
                })
    elapsed = time.time() - started
    predictions = pd.DataFrame(rows)
    metadata = {
        "parse_errors": parse_errors,
        "score_time_seconds": elapsed,
        "rows_per_second": len(predictions) / elapsed if elapsed > 0 else None,
    }
    return predictions, metadata


def evaluate_vllm_model(
    *,
    model_name: str,
    adapter_dir: Path,
    records: SplitRecords,
    max_new_tokens: int,
    rating_min: int,
    rating_max: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int | None,
    batch_size: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "tokenizer": adapter_dir.as_posix(),
        "dtype": dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": False,
        "enable_lora": True,
        "max_lora_rank": 16,
        "max_model_len": max_model_len,
    }
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    lora_request = LoRARequest(
        lora_name=adapter_dir.parent.name,
        lora_int_id=1,
        lora_path=adapter_dir.as_posix(),
    )

    prompts = records.frame["prompt"].tolist()
    started = time.time()
    if batch_size is None:
        outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    else:
        outputs = []
        for start in range(0, len(prompts), batch_size):
            outputs.extend(
                llm.generate(
                    prompts[start:start + batch_size],
                    sampling,
                    lora_request=lora_request,
                )
            )
    elapsed = time.time() - started

    rows = []
    parse_errors = 0
    for (_, row), output in zip(records.frame.iterrows(), outputs, strict=True):
        text = output.outputs[0].text if output.outputs else ""
        rating = parse_rating(text, rating_min=rating_min, rating_max=rating_max)
        if rating is None:
            parse_errors += 1
            score = 0.0
        else:
            score = rating_to_score(rating, rating_min=rating_min, rating_max=rating_max)
        rows.append({
            "dataset": row["dataset"],
            "index": row["index"],
            "label": int(row["label"]),
            "score": float(score),
            "rating": rating,
            "parse_error": rating is None,
            "format_valid": structured_completion_match(text) is not None,
            "generation": text,
        })

    predictions = pd.DataFrame(rows)
    metadata = {
        "parse_errors": parse_errors,
        "score_time_seconds": elapsed,
        "rows_per_second": len(predictions) / elapsed if elapsed > 0 else None,
    }
    return predictions, metadata


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    y_pred = (y_score >= threshold).astype(int)
    tp, tn, fp, fn = confusion(y_true, y_pred)
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    balanced_accuracy = None if recall is None or fpr is None else (recall + (1.0 - fpr)) / 2.0
    return {
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "fpr": fpr,
    }


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    from sklearn.metrics import roc_auc_score

    metrics = threshold_metrics(y_true, y_score, threshold)
    auroc = None
    if np.unique(y_true).size >= 2:
        auroc = float(roc_auc_score(y_true, y_score))
    return {**metrics, "auroc": auroc}


def macro_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | None]:
    per_dataset = [
        binary_metrics(group["label"].to_numpy(), group["score"].to_numpy(), threshold)
        for _, group in frame.groupby("dataset", sort=True)
    ]
    out = {}
    for key in ["balanced_accuracy", "auroc", "recall", "fpr"]:
        values = [m[key] for m in per_dataset if m[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def per_dataset_table(frame: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for dataset, group in frame.groupby("dataset", sort=True):
        metrics = binary_metrics(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            threshold,
        )
        rows.append({"dataset": dataset, "n": int(len(group)), **metrics})
    return rows


def write_predictions(path: Path, frame: pd.DataFrame, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "index": frame["index"],
        "deceptive": frame["score"] >= threshold,
        "score": frame["score"],
        "label": frame["label"],
        "dataset": frame["dataset"],
    }).to_csv(path, index=False)


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
    generated_methods = {str(record.get("method", "-")) for record in records}

    rows: list[tuple[str, str]] = []
    for record in records:
        metrics = record.get("metrics", {})
        timing = record.get("timing", {})
        submitted_at = fmt_submitted_at(record.get("submitted_at"))
        rows.append((
            submitted_at,
            "| "
            f"{submitted_at} | "
            f"{record.get('method', '-')} | "
            f"{fmt_metric(metrics.get('auroc'))} | "
            f"{fmt_metric(metrics.get('balanced_accuracy'))} | "
            f"{fmt_metric(metrics.get('recall'))} | "
            f"{fmt_metric(metrics.get('fpr'))} | "
            f"{fmt_seconds(timing.get('score_seconds'))} | "
            f"{fmt_rate(timing.get('rows_per_second'))} |",
        ))

    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if not line.startswith("| "):
                continue
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) != 8 or parts[0] in {"submitted_at", "---"}:
                continue
            if parts[1] in generated_methods:
                continue
            rows.append((parts[0], line))

    rows.sort(key=lambda item: item[0], reverse=True)

    lines = [
        "# Black-Box Experiment Leaderboard",
        "",
        "Timing is scoring-only wall time: it excludes vLLM startup/model load/compile and dataset preparation.",
        "Submitted timestamps are UTC.",
        "Rows with cached text-probe components report the logits scoring wall time; CPU n-gram inference is negligible.",
        "",
        "| submitted_at | method | AUROC | bal_acc | recall | FPR | score_time | rows/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(line for _, line in rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def load_training_config(adapter_dir: Path) -> dict[str, Any]:
    result_path = adapter_dir.parent / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        config = result.get("config")
        if isinstance(config, dict):
            return config

    config_path = adapter_dir.parent / "config.json"
    if config_path.exists():
        metadata = json.loads(config_path.read_text())
        config = metadata.get("config")
        if isinstance(config, dict):
            return config

    raise FileNotFoundError(
        f"could not find training config next to adapter {adapter_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Saved PEFT adapter directory.",
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--global-limit", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--leaderboard", type=Path, default=ROOT / "results/blackbox/leaderboard.md")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    return parser.parse_args()


def cfg_get(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = value[part]
    return value


def main() -> None:
    args = parse_args()
    adapter_dir = args.adapter_dir.resolve()
    config = load_training_config(adapter_dir)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else adapter_dir.parent / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading {args.split} split", flush=True)
    records = load_split(
        args.split,
        args.splits_dir.resolve(),
        prompt_template=str(cfg_get(config, "judge.prompt")),
        tokenizer=tokenizer,
        max_prompt_chars=int(cfg_get(config, "judge.max_prompt_chars")),
        context_truncation=str(cfg_get(config, "judge.context_truncation")),
        include_reasoning=bool(cfg_get(config, "judge.include_reasoning")),
        reasoning_max_chars=int(cfg_get(config, "judge.reasoning_max_chars")),
        reasoning_truncation=str(cfg_get(config, "judge.reasoning_truncation")),
        enable_thinking=bool(cfg_get(config, "judge.enable_thinking")),
    )
    records = apply_global_limit(records, args.global_limit)
    print(
        f"{args.split} rows={len(records.frame)} datasets={len(records.dataset_names)} "
        f"positives={int(records.frame['label'].sum())}",
        flush=True,
    )

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(cfg_get(config, "training.max_completion_length"))
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(cfg_get(config, "evaluation.batch_size"))
    )
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(cfg_get(config, "scoring.baseline_threshold"))
    )
    max_model_len = (
        args.max_model_len
        if args.max_model_len is not None
        else int(cfg_get(config, "training.max_prompt_length")) + max_new_tokens
    )

    print(f"evaluating split with {args.backend}", flush=True)
    if args.backend == "vllm":
        predictions, eval_meta = evaluate_vllm_model(
            model_name=str(config["model"]),
            adapter_dir=adapter_dir,
            records=records,
            max_new_tokens=max_new_tokens,
            rating_min=1,
            rating_max=7,
            dtype=str(args.dtype),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=max_model_len,
            max_num_seqs=args.max_num_seqs,
            batch_size=args.batch_size,
        )
    else:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        import torch

        print("loading base model and adapter", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(config["model"]),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.config.use_cache = True
        if torch.cuda.is_available():
            model = model.to("cuda")
        predictions, eval_meta = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            records=records,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            rating_min=1,
            rating_max=7,
        )
    metrics = macro_metrics(predictions, threshold=threshold)

    prediction_paths: dict[str, str] = {}
    for dataset, group in predictions.groupby("dataset", sort=True):
        pred_path = output_dir / "predictions" / f"{dataset.replace('/', '__')}.csv"
        write_predictions(pred_path, group, threshold)
        prediction_paths[dataset] = pred_path.as_posix()

    predictions_path = output_dir / "predictions.csv"
    generations_path = output_dir / "generations.jsonl"
    result_path = output_dir / "result.json"
    write_predictions(predictions_path, predictions, threshold)
    generations_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in predictions.to_dict("records")
        )
        + "\n"
    )

    datasets = [
        DatasetResult(
            dataset=dataset,
            n=int(len(group)),
            metrics=macro_metrics(group, threshold=threshold),
            predictions_path=prediction_paths[dataset],
        )
        for dataset, group in predictions.groupby("dataset", sort=True)
    ]
    result = {
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": str(config["method"]),
        "split": args.split,
        "n": int(len(predictions)),
        "n_prompt_evaluations": int(len(predictions)),
        "metrics": metrics,
        "timing": {
            "score_seconds": eval_meta["score_time_seconds"],
            "rows_per_second": eval_meta["rows_per_second"],
            "prompt_evaluations_per_second": eval_meta["rows_per_second"],
            "note": "excludes model startup and dataset preparation",
        },
        "config": {
            "method": str(config["method"]),
            "split": args.split,
            "output_dir": str(output_dir.parent.parent),
            "splits_dir": str(args.splits_dir),
            "scoring": {"threshold": threshold},
            "adapter_dir": adapter_dir.as_posix(),
            "evaluator_backend": args.backend,
            "dtype": str(args.dtype),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "max_model_len": max_model_len,
            "max_new_tokens": max_new_tokens,
            "max_num_seqs": args.max_num_seqs,
            "training_config": config,
        },
        "datasets": [asdict(dataset) for dataset in datasets],
        "run_dir": output_dir.as_posix(),
        "result_path": result_path.as_posix(),
        "parse_errors": eval_meta["parse_errors"],
        "generations_path": generations_path.as_posix(),
        "flat_predictions_path": predictions_path.as_posix(),
        "adapter_path": adapter_dir.as_posix(),
        "threshold": threshold,
        "per_dataset": per_dataset_table(predictions, threshold),
    }
    result_path.write_text(json.dumps(result, indent=2))
    render_leaderboard(output_dir.parent.parent, args.leaderboard.resolve())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
