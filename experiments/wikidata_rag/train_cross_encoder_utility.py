#!/usr/bin/env python3
"""Fine-tune a pretrained cross-encoder to rank facts against no evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.wikidata_rag.train_utility_retriever import (
    candidate_indices,
    choose_threshold,
    group_bucket,
    safe_metric,
    selection_report,
    serializable_report,
)

NO_EVIDENCE_TEXT = "No external evidence is available."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def serialize_query(row: dict[str, Any], query_mode: str = "full") -> str:
    """Serialize only observable claim context and frozen-reader state."""
    score = float(row["empty_score"])
    context = (
        f"Question: {row['question']}\n"
        f"Assistant answer: {row['answer_full']}"
    )
    if query_mode == "no_score":
        return context
    if query_mode == "score_bucket":
        if score < 0.1:
            bucket = "very likely honest"
        elif score < 0.4:
            bucket = "likely honest"
        elif score <= 0.6:
            bucket = "uncertain"
        elif score <= 0.9:
            bucket = "likely deceptive"
        else:
            bucket = "very likely deceptive"
        return f"{context}\nCurrent detector state: {bucket}"
    if query_mode == "full":
        return f"{context}\nCurrent detector deception score: {score:.4f}"
    raise ValueError(f"Unknown query mode: {query_mode}")


def serialize_candidate(candidate: dict[str, Any]) -> str:
    return (
        f"Evidence subject: {candidate['subject']}\n"
        f"Evidence statement: {candidate['fact']}"
    )


def split_row_ids(
    training_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, list[int]]:
    rows = training_rows + validation_rows
    groups = [str(row["question_group"]) for row in rows]
    training_groups = set(groups[: len(training_rows)])
    buckets = [group_bucket(group) for group in groups]
    frozen = list(range(len(training_rows), len(rows)))
    return {
        "train": [
            index for index in range(len(training_rows))
            if buckets[index] not in {0, 1}
        ],
        "calibration": [
            index for index in range(len(training_rows)) if buckets[index] == 1
        ],
        "internal_test": [
            index for index in range(len(training_rows)) if buckets[index] == 0
        ],
        "frozen": frozen,
        "frozen_novel": [
            index for index in frozen if groups[index] not in training_groups
        ],
        "frozen_seen": [
            index for index in frozen if groups[index] in training_groups
        ],
    }


def row_slices(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    result = []
    start = 0
    for row in rows:
        end = start + len(row["candidates"])
        result.append((start, end))
        start = end
    return result


def clipped_logit(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def candidate_target(
    row: dict[str, Any], candidate: dict[str, Any], target_name: str
) -> float:
    if target_name in {"utility", "controlled_utility"}:
        return float(candidate[target_name])
    baseline_correct = (
        (float(row["empty_score"]) >= 0.5) == bool(row["label"])
    )
    candidate_correct = (
        (float(candidate["score"]) >= 0.5) == bool(row["label"])
    )
    if target_name == "binary_utility":
        return float(candidate_correct) - float(baseline_correct)
    if target_name == "controlled_binary_utility":
        shuffled_correct = (
            (float(row["shuffled_control"]["score"]) >= 0.5)
            == bool(row["label"])
        )
        return float(candidate_correct) - float(shuffled_correct)
    if target_name == "score_delta":
        return clipped_logit(candidate["score"]) - clipped_logit(row["empty_score"])
    if target_name in {"semantic_decisive", "semantic_relevant"}:
        label = candidate.get("semantic_label")
        if label is None:
            return float("nan")
        positive = {"decisive"}
        if target_name == "semantic_relevant":
            positive.add("relevant_insufficient")
        return float(label in positive)
    raise ValueError(f"Unknown target: {target_name}")


def candidate_targets(rows: list[dict[str, Any]], target_name: str) -> np.ndarray:
    return np.asarray([
        candidate_target(row, candidate, target_name)
        for row in rows for candidate in row["candidates"]
    ], dtype=np.float32)


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], row_ids: list[int]) -> None:
        self.rows = rows
        self.row_ids = row_ids

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[self.row_ids[index]]


def collate_rows(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def group_weights(rows: list[dict[str, Any]], row_ids: Iterable[int]) -> dict[str, float]:
    counts = Counter(str(rows[row_id]["question_group"]) for row_id in row_ids)
    return {group: 1.0 / count for group, count in counts.items()}


def flatten_batch(
    batch: list[dict[str, Any]], target_name: str, query_mode: str = "full"
) -> tuple[list[str], list[str], list[torch.Tensor]]:
    queries: list[str] = []
    documents: list[str] = []
    targets: list[torch.Tensor] = []
    for row in batch:
        query = serialize_query(row, query_mode)
        row_targets = [
            candidate_target(row, candidate, target_name)
            for candidate in row["candidates"]
        ]
        row_documents = [serialize_candidate(candidate) for candidate in row["candidates"]]
        row_documents.append(NO_EVIDENCE_TEXT)
        row_targets.append(0.0)
        queries.extend([query] * len(row_documents))
        documents.extend(row_documents)
        targets.append(torch.tensor(row_targets, dtype=torch.float32))
    return queries, documents, targets


def pairwise_utility_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_scale: float,
    minimum_gain: float,
    regression_weight: float,
    loss_mode: str = "pairwise",
    listwise_temperature: float = 0.1,
) -> torch.Tensor:
    """Combine robust utility regression with within-row pairwise ordering."""
    finite = torch.isfinite(targets)
    predictions = predictions[finite]
    targets = targets[finite]
    if not len(targets):
        return predictions.new_zeros(())
    if loss_mode == "hard_listwise":
        candidate_targets_only = targets[:-1]
        if not len(candidate_targets_only):
            return predictions.sum() * 0.0
        best = int(torch.argmax(candidate_targets_only))
        if float(candidate_targets_only[best]) <= minimum_gain:
            best = len(targets) - 1
        return F.cross_entropy(
            predictions.reshape(1, -1),
            torch.tensor([best], dtype=torch.long, device=predictions.device),
        )
    if loss_mode == "soft_listwise":
        distribution = torch.softmax(targets / listwise_temperature, dim=0)
        return -(distribution * torch.log_softmax(predictions, dim=0)).sum()
    if loss_mode != "pairwise":
        raise ValueError(f"Unknown loss mode: {loss_mode}")
    normalized = torch.clamp(targets / target_scale, -5.0, 5.0)
    regression = F.smooth_l1_loss(predictions, normalized, reduction="mean", beta=0.5)
    target_difference = targets[:, None] - targets[None, :]
    prediction_difference = predictions[:, None] - predictions[None, :]
    upper = torch.triu(
        torch.ones_like(target_difference, dtype=torch.bool), diagonal=1
    )
    material = upper & (target_difference.abs() > minimum_gain)
    if material.any():
        signs = target_difference[material].sign()
        ranking = F.softplus(-signs * prediction_difference[material]).mean()
    else:
        ranking = predictions.new_zeros(())
    return regression_weight * regression + ranking


def model_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    queries: list[str],
    documents: list[str],
    *,
    device: torch.device,
    max_length: int,
    sequence_batch_size: int,
    use_bf16: bool,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(queries), sequence_batch_size):
        encoded = tokenizer(
            queries[start : start + sequence_batch_size],
            documents[start : start + sequence_batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 else nullcontext()
        )
        with context:
            logits = model(**encoded).logits.reshape(-1)
        parts.append(logits.float())
    return torch.cat(parts)


def train_epoch(
    model: torch.nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    device: torch.device,
    target_name: str,
    target_scale: float,
    minimum_gain: float,
    regression_weight: float,
    max_length: int,
    sequence_batch_size: int,
    use_bf16: bool,
    weights: dict[str, float],
    gradient_clip: float,
    loss_mode: str = "pairwise",
    listwise_temperature: float = 0.1,
    query_mode: str = "full",
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    row_count = 0
    started = time.monotonic()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        queries, documents, target_parts = flatten_batch(batch, target_name, query_mode)
        flat_predictions = model_scores(
            model, tokenizer, queries, documents, device=device,
            max_length=max_length, sequence_batch_size=sequence_batch_size,
            use_bf16=use_bf16,
        )
        losses = []
        offset = 0
        for row, targets in zip(batch, target_parts, strict=True):
            targets = targets.to(device)
            end = offset + len(targets)
            loss = pairwise_utility_loss(
                flat_predictions[offset:end], targets,
                target_scale=target_scale, minimum_gain=minimum_gain,
                regression_weight=regression_weight,
                loss_mode=loss_mode,
                listwise_temperature=listwise_temperature,
            )
            weight = weights[str(row["question_group"])]
            losses.append(loss * weight)
            offset = end
        denominator = sum(weights[str(row["question_group"])] for row in batch)
        loss = torch.stack(losses).sum() / denominator
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach()) * len(batch)
        row_count += len(batch)
    return {
        "loss": loss_sum / max(1, row_count),
        "rows": row_count,
        "seconds": time.monotonic() - started,
    }


@torch.no_grad()
def predict_rows(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    row_batch_size: int,
    sequence_batch_size: int,
    use_bf16: bool,
    query_mode: str = "full",
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, float]]:
    model.eval()
    predictions = []
    slices = []
    started = time.monotonic()
    for batch_start in range(0, len(rows), row_batch_size):
        batch = rows[batch_start : batch_start + row_batch_size]
        queries, documents, targets = flatten_batch(
            batch, "controlled_utility", query_mode
        )
        scores = model_scores(
            model, tokenizer, queries, documents, device=device,
            max_length=max_length, sequence_batch_size=sequence_batch_size,
            use_bf16=use_bf16,
        ).cpu().numpy()
        offset = 0
        for target in targets:
            count = len(target) - 1
            no_evidence_score = float(scores[offset + count])
            start = len(predictions)
            predictions.extend(
                (scores[offset : offset + count] - no_evidence_score).tolist()
            )
            slices.append((start, len(predictions)))
            offset += count + 1
    seconds = time.monotonic() - started
    return np.asarray(predictions, dtype=np.float32), slices, {
        "seconds": seconds,
        "candidate_pairs": len(predictions),
        "pairs_per_second": (len(predictions) + len(rows)) / max(seconds, 1e-9),
    }


def candidate_quality(
    targets: np.ndarray,
    predictions: np.ndarray,
    slices: list[tuple[int, int]],
    row_ids: list[int],
    *,
    minimum_gain: float,
) -> dict[str, float | int]:
    indices = candidate_indices(slices, row_ids)
    finite = np.isfinite(targets[indices])
    indices = indices[finite]
    labels = (targets[indices] > minimum_gain).astype(np.int8)
    return {
        "candidates": int(len(indices)),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "auroc": safe_metric(roc_auc_score, labels, predictions[indices]),
        "average_precision": safe_metric(
            average_precision_score, labels, predictions[indices]
        ),
    }


def transform_policy_scores(
    predictions: np.ndarray,
    slices: list[tuple[int, int]],
    score_mode: str,
) -> np.ndarray:
    """Calibrate cross-row confidence while preserving within-row ranking."""
    if score_mode == "margin":
        return predictions
    transformed = predictions.copy()
    for start, end in slices:
        values = predictions[start:end]
        if score_mode == "count_adjusted":
            transformed[start:end] = values - math.log(max(1, len(values)))
        elif score_mode == "softmax_logprob":
            augmented = np.concatenate((np.zeros(1, dtype=values.dtype), values))
            maximum = float(augmented.max())
            log_normalizer = maximum + math.log(
                float(np.exp(augmented - maximum).sum())
            )
            transformed[start:end] = values - log_normalizer
        else:
            raise ValueError(f"Unknown score mode: {score_mode}")
    return transformed


def evaluate_checkpoint(
    rows: list[dict[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
    slices: list[tuple[int, int]],
    splits: dict[str, list[int]],
    *,
    minimum_gain: float,
    minimum_precision: float,
    minimum_emitted: int,
    score_mode: str = "margin",
) -> dict[str, Any]:
    predictions = transform_policy_scores(predictions, slices, score_mode)
    threshold, calibration = choose_threshold(
        rows, predictions, slices, splits["calibration"],
        minimum_gain=minimum_gain, minimum_precision=minimum_precision,
        minimum_emitted=minimum_emitted, selection_objective="balanced_accuracy",
    )

    def report(name: str) -> dict[str, Any]:
        if not splits[name]:
            return {
                "rows": 0,
                "emitted": 0,
                "coverage": 0.0,
                "controlled_positive": 0,
                "controlled_positive_precision": 0.0,
                "raw_harmful": 0,
                "raw_harm_rate": 0.0,
                "semantic_known": 0,
                "semantic_decisive": 0,
                "semantic_decisive_precision": 0.0,
                "balanced_raw_gain": 0.0,
                "balanced_controlled_gain": 0.0,
                "mean_raw_gain_emitted": 0.0,
                "mean_controlled_gain_emitted": 0.0,
                "balanced_accuracy_delta": None,
                "auroc_delta": None,
                "baseline_metrics": {},
                "selected_metrics": {},
                "matched_shuffled_metrics": {},
            }
        return serializable_report(selection_report(
            rows, predictions, slices, splits[name], threshold,
            minimum_gain=minimum_gain,
        ))

    return {
        "score_mode": score_mode,
        "threshold": threshold,
        "calibration": serializable_report(calibration),
        "internal_test": report("internal_test"),
        "frozen_validation": report("frozen"),
        "frozen_novel": report("frozen_novel"),
        "frozen_seen": report("frozen_seen"),
        "candidate_quality": {
            name: candidate_quality(
                targets, predictions, slices, splits[name],
                minimum_gain=minimum_gain,
            )
            for name in (
                "calibration", "internal_test", "frozen",
                "frozen_novel", "frozen_seen",
            )
        },
    }


def checkpoint_key(
    report: dict[str, Any], selection_mode: str = "calibration"
) -> tuple[float, float, float, float]:
    calibration = report["calibration"]
    if selection_mode == "robust":
        internal = report["internal_test"]
        calibration_ba = float(calibration["balanced_accuracy_delta"] or 0.0)
        internal_ba = float(internal["balanced_accuracy_delta"] or 0.0)
        return (
            min(calibration_ba, internal_ba),
            (calibration_ba + internal_ba) / 2.0,
            min(
                float(calibration["balanced_controlled_gain"]),
                float(internal["balanced_controlled_gain"]),
            ),
            (
                float(report["candidate_quality"]["calibration"]["average_precision"])
                + float(report["candidate_quality"]["internal_test"]["average_precision"])
            ) / 2.0,
        )
    if selection_mode != "calibration":
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    return (
        float(calibration["balanced_accuracy_delta"] or 0.0),
        float(calibration["balanced_controlled_gain"]),
        float(calibration["balanced_raw_gain"]),
        float(report["candidate_quality"]["calibration"]["average_precision"]),
    )


def linear_schedule(optimizer: torch.optim.Optimizer, total_steps: int) -> Any:
    warmup = max(1, round(0.1 * total_steps))

    def multiplier(step: int) -> float:
        if step < warmup:
            return max(1e-3, (step + 1) / warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def load_model(model_name: str, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if int(model.config.num_labels) != 1:
        raise ValueError(f"Expected one reranker logit, got {model.config.num_labels}")
    model.to(device)
    return model, tokenizer


def freeze_bottom_layers(model: torch.nn.Module, count: int) -> int:
    """Freeze embeddings and the requested number of bottom encoder blocks."""
    if count <= 0:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    base = getattr(model, "bert", None)
    if base is None or not hasattr(base, "encoder"):
        raise ValueError("Layer freezing currently expects a BERT-style reranker")
    for parameter in base.embeddings.parameters():
        parameter.requires_grad = False
    layers = list(base.encoder.layer)
    if count > len(layers):
        raise ValueError(f"Cannot freeze {count} layers from a {len(layers)}-layer model")
    for layer in layers[:count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def target_training_rows(
    rows: list[dict[str, Any]], row_ids: list[int], target_name: str
) -> list[int]:
    """Exclude wholly unlabeled rows from semantic auxiliary objectives."""
    if not target_name.startswith("semantic_"):
        return row_ids
    return [
        row_id for row_id in row_ids
        if any(candidate.get("semantic_label") is not None for candidate in rows[row_id]["candidates"])
    ]


def save_best(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    metadata: dict[str, Any],
) -> int:
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    (output_dir / "selected_checkpoint.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())


def parse_floats(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--targets", nargs="+", choices=(
            "utility", "controlled_utility", "binary_utility",
            "controlled_binary_utility", "score_delta",
            "semantic_decisive", "semantic_relevant",
        ),
        default=["controlled_utility", "utility"],
    )
    parser.add_argument(
        "--loss-modes", nargs="+",
        choices=("pairwise", "hard_listwise", "soft_listwise"),
        default=["pairwise"],
    )
    parser.add_argument("--freeze-bottom-layers", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--query-modes", nargs="+", choices=("full", "score_bucket", "no_score"),
        default=["full"],
    )
    parser.add_argument(
        "--score-modes", nargs="+",
        choices=("margin", "count_adjusted", "softmax_logprob"),
        default=["margin", "count_adjusted", "softmax_logprob"],
    )
    parser.add_argument("--learning-rates", nargs="+", default=["1e-5", "3e-5"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--row-batch-size", type=int, default=8)
    parser.add_argument("--sequence-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    parser.add_argument("--minimum-precision", type=float, default=0.6)
    parser.add_argument("--minimum-emitted", type=int, default=5)
    parser.add_argument("--regression-weight", type=float, default=0.25)
    parser.add_argument("--listwise-temperature", type=float, default=0.1)
    parser.add_argument(
        "--selection-mode", choices=("calibration", "robust"),
        default="calibration",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-training-rows", type=int)
    parser.add_argument("--limit-validation-rows", type=int)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    training_rows = load_jsonl(args.input)
    validation_rows = load_jsonl(args.validation_input)
    if args.limit_training_rows:
        training_rows = training_rows[: args.limit_training_rows]
    if args.limit_validation_rows:
        validation_rows = validation_rows[: args.limit_validation_rows]
    rows = training_rows + validation_rows
    splits = split_row_ids(training_rows, validation_rows)
    slices = row_slices(rows)
    learning_rates = parse_floats(args.learning_rates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    if not splits["train"] or not splits["calibration"] or not splits["internal_test"]:
        raise SystemExit("The grouped smoke subset must populate train/calibration/test")

    all_reports: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_metadata: dict[str, Any] | None = None
    selected_model_bytes = 0

    model, tokenizer = load_model(args.model_name, device)
    zero_predictions, zero_slices, zero_timing = predict_rows(
        model, tokenizer, rows, device=device, max_length=args.max_length,
        row_batch_size=args.row_batch_size,
        sequence_batch_size=args.sequence_batch_size, use_bf16=use_bf16,
    )
    zero_targets = candidate_targets(rows, "controlled_utility")
    zero_report = evaluate_checkpoint(
        rows, zero_targets, zero_predictions, zero_slices, splits,
        minimum_gain=args.minimum_gain, minimum_precision=args.minimum_precision,
        minimum_emitted=args.minimum_emitted,
    )
    zero_report.update({"stage": "zero_shot", "timing": zero_timing})
    all_reports.append(zero_report)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for target_name in args.targets:
        targets = candidate_targets(rows, target_name)
        training_row_ids = target_training_rows(rows, splits["train"], target_name)
        train_candidates = candidate_indices(slices, training_row_ids)
        target_scale = max(float(np.nanstd(targets[train_candidates])), 1e-3)
        for loss_mode in args.loss_modes:
            for query_mode in args.query_modes:
                for frozen_layers in args.freeze_bottom_layers:
                    for learning_rate in learning_rates:
                        # Match initialization, shuffling, and dropout noise across
                        # configurations so the sweep measures the ablation itself.
                        random.seed(args.seed)
                        np.random.seed(args.seed)
                        torch.manual_seed(args.seed)
                        model, tokenizer = load_model(args.model_name, device)
                        trainable_parameters = freeze_bottom_layers(model, frozen_layers)
                        dataset = RowDataset(rows, training_row_ids)
                        generator = torch.Generator().manual_seed(args.seed)
                        loader = DataLoader(
                            dataset, batch_size=args.row_batch_size, shuffle=True,
                            collate_fn=collate_rows, num_workers=0, generator=generator,
                        )
                        optimizer = torch.optim.AdamW(
                            (parameter for parameter in model.parameters() if parameter.requires_grad),
                            lr=learning_rate, weight_decay=args.weight_decay,
                        )
                        scheduler = linear_schedule(optimizer, args.epochs * len(loader))
                        weights = group_weights(rows, training_row_ids)
                        for epoch in range(1, args.epochs + 1):
                            train_report = train_epoch(
                                model, tokenizer, loader, optimizer, scheduler,
                                device=device, target_name=target_name,
                                target_scale=target_scale, minimum_gain=args.minimum_gain,
                                regression_weight=args.regression_weight,
                                max_length=args.max_length,
                                sequence_batch_size=args.sequence_batch_size,
                                use_bf16=use_bf16, weights=weights,
                                gradient_clip=args.gradient_clip,
                                loss_mode=loss_mode,
                                listwise_temperature=args.listwise_temperature,
                                query_mode=query_mode,
                            )
                            predictions, predicted_slices, timing = predict_rows(
                                model, tokenizer, rows, device=device,
                                max_length=args.max_length,
                                row_batch_size=args.row_batch_size,
                                sequence_batch_size=args.sequence_batch_size,
                                use_bf16=use_bf16, query_mode=query_mode,
                            )
                            for score_mode in args.score_modes:
                                report = evaluate_checkpoint(
                                    rows, targets, predictions, predicted_slices, splits,
                                    minimum_gain=args.minimum_gain,
                                    minimum_precision=args.minimum_precision,
                                    minimum_emitted=args.minimum_emitted,
                                    score_mode=score_mode,
                                )
                                metadata = {
                                    "stage": "fine_tuned", "target": target_name,
                                    "loss_mode": loss_mode, "query_mode": query_mode,
                                    "score_mode": score_mode,
                                    "frozen_bottom_layers": frozen_layers,
                                    "trainable_parameters": trainable_parameters,
                                    "training_rows": len(training_row_ids),
                                    "learning_rate": learning_rate, "epoch": epoch,
                                    "target_scale": target_scale,
                                }
                                report.update(metadata | {
                                    "training": train_report, "timing": timing,
                                })
                                all_reports.append(report)
                                key = checkpoint_key(report, args.selection_mode)
                                print(json.dumps({
                                    **metadata,
                                    "calibration": report["calibration"],
                                    "internal_test": report["internal_test"],
                                    "frozen_validation": report["frozen_validation"],
                                    "frozen_novel": report["frozen_novel"],
                                    "checkpoint_key": key,
                                }), flush=True)
                                if best_key is None or key > best_key:
                                    best_key = key
                                    best_metadata = metadata | {
                                        "threshold": report["threshold"],
                                        "checkpoint_key": list(key),
                                    }
                                    selected_model_bytes = save_best(
                                        model, tokenizer, args.output_dir, best_metadata
                                    )
                        del model, optimizer, scheduler
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

    report = {
        "model_name": args.model_name,
        "training_rows": len(training_rows),
        "validation_rows": len(validation_rows),
        "unique_training_groups": len({row["question_group"] for row in training_rows}),
        "split_rows": {key: len(value) for key, value in splits.items()},
        "targets": args.targets,
        "loss_modes": args.loss_modes,
        "freeze_bottom_layers": args.freeze_bottom_layers,
        "query_modes": args.query_modes,
        "score_modes": args.score_modes,
        "learning_rates": learning_rates,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "minimum_gain": args.minimum_gain,
        "minimum_precision": args.minimum_precision,
        "minimum_emitted": args.minimum_emitted,
        "regression_weight": args.regression_weight,
        "listwise_temperature": args.listwise_temperature,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "row_batch_size": args.row_batch_size,
        "sequence_batch_size": args.sequence_batch_size,
        "selection_mode": args.selection_mode,
        "device": str(device),
        "bf16": use_bf16,
        "selected": best_metadata,
        "selected_model_bytes": selected_model_bytes,
        "checkpoints": all_reports,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "selected": best_metadata,
        "selected_model_bytes": selected_model_bytes,
        "report": (args.output_dir / "report.json").as_posix(),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
