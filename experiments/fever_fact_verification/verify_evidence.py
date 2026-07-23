#!/usr/bin/env python3
"""Score Wikipedia sentences with a FEVER-trained three-way NLI verifier."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.core import aggregate_evidence


DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label_indices(model: Any) -> dict[str, int]:
    labels = {str(value).lower(): int(key) for key, value in model.config.id2label.items()}
    aliases = {
        "entailment": ("entailment", "supports", "label_0"),
        "neutral": ("neutral", "not enough info", "nei", "label_1"),
        "contradiction": ("contradiction", "refutes", "label_2"),
    }
    output = {}
    for target, candidates in aliases.items():
        match = next((labels[item] for item in candidates if item in labels), None)
        if match is None:
            raise ValueError(f"cannot identify {target} in labels {labels}")
        output[target] = match
    return output


def make_pairs(
    rows: list[dict[str, Any]],
    *,
    max_candidates: int,
    shuffle_control: bool,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[int, int]], list[list[dict[str, Any]]]]:
    evidence_lists = [list(row.get("passages") or [])[:max_candidates] for row in rows]
    if shuffle_control and evidence_lists:
        rng = random.Random(seed)
        order = list(range(len(rows)))
        for _ in range(100):
            rng.shuffle(order)
            if all(
                rows[source]["dataset"] != rows[target]["dataset"]
                for target, source in enumerate(order)
            ):
                break
        else:
            order = order[len(order) // 2 :] + order[: len(order) // 2]
        evidence_lists = [evidence_lists[source] for source in order]

    pairs: list[tuple[str, str]] = []
    locations: list[tuple[int, int]] = []
    for row_index, (row, passages) in enumerate(zip(rows, evidence_lists)):
        for passage_index, passage in enumerate(passages):
            premise = f"{passage.get('title', '')}. {passage.get('text', '')}"
            pairs.append((premise, str(row["proposition"])))
            locations.append((row_index, passage_index))
    return pairs, locations, evidence_lists


@torch.inference_mode()
def rerank_evidence(
    rows: list[dict[str, Any]],
    evidence_lists: list[list[dict[str, Any]]],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    top_n: int,
    minimum_score: float,
) -> list[list[dict[str, Any]]]:
    """Apply a pointwise relevance cross-encoder before three-way NLI."""
    pairs = []
    locations = []
    for row_index, (row, passages) in enumerate(zip(rows, evidence_lists)):
        for passage_index, passage in enumerate(passages):
            evidence = f"{passage.get('title', '')}. {passage.get('text', '')}"
            pairs.append((str(row["proposition"]), evidence))
            locations.append((row_index, passage_index))
    device = next(model.parameters()).device
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits.float()
        if logits.shape[-1] == 1:
            # CrossEncoder models expose a single relevance logit. Convert it
            # to a probability so --minimum-reranker-score has stable meaning;
            # the sigmoid is monotonic and therefore leaves ranking unchanged.
            batch_scores = torch.sigmoid(logits[:, 0])
        else:
            batch_scores = torch.softmax(logits, dim=-1)[:, -1]
        scores.extend(float(value) for value in batch_scores.cpu())

    scored: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (row_index, passage_index), score in zip(locations, scores):
        scored[row_index].append({
            **evidence_lists[row_index][passage_index],
            "reranker_score": score,
        })
    output = []
    for row_index in range(len(rows)):
        ranked = sorted(
            scored[row_index], key=lambda item: float(item["reranker_score"]), reverse=True
        )
        output.append([
            item for item in ranked[:top_n]
            if float(item["reranker_score"]) >= minimum_score
        ])
    return output


@torch.inference_mode()
def score_pairs(
    pairs: list[tuple[str, str]],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    indices: dict[str, int],
) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    output = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu()
        for scores in probabilities:
            output.append({name: float(scores[index]) for name, index in indices.items()})
    return output


def run_condition(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    reranker_model: Any,
    reranker_tokenizer: Any,
    batch_size: int,
    max_length: int,
    max_candidates: int,
    rerank_top_n: int,
    minimum_reranker_score: float,
    minimum_confidence: float,
    top_k: int,
    shuffle_control: bool,
    seed: int,
) -> list[dict[str, Any]]:
    _, _, evidence_lists = make_pairs(
        rows,
        max_candidates=max_candidates,
        shuffle_control=shuffle_control,
        seed=seed,
    )
    evidence_lists = rerank_evidence(
        rows,
        evidence_lists,
        model=reranker_model,
        tokenizer=reranker_tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        top_n=rerank_top_n,
        minimum_score=minimum_reranker_score,
    )
    pairs = []
    locations = []
    for row_index, (row, passages) in enumerate(zip(rows, evidence_lists)):
        for passage_index, passage in enumerate(passages):
            premise = f"{passage.get('title', '')}. {passage.get('text', '')}"
            pairs.append((premise, str(row["proposition"])))
            locations.append((row_index, passage_index))
    scores = score_pairs(
        pairs,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        indices=label_indices(model),
    )
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (row_index, passage_index), score in zip(locations, scores):
        by_row[row_index].append({**evidence_lists[row_index][passage_index], **score})
    output = []
    for row_index, row in enumerate(rows):
        aggregate = aggregate_evidence(
            by_row[row_index],
            minimum_confidence=minimum_confidence,
            top_k=top_k,
        )
        identity = {key: row[key] for key in (
                "dataset", "index", "claim_index", "label", "quote",
                "proposition", "teacher_assessment",
            )}
        if "question" in row:
            identity["question"] = row["question"]
        identity["retrieval_error"] = row.get("error")
        identity["candidate_count"] = len(by_row[row_index])
        output.append({
            **identity,
            "condition": "shuffled" if shuffle_control else "real",
            **aggregate,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--label-order",
        help="Comma-separated verifier labels by logit index, e.g. neutral,contradiction,entailment",
    )
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--rerank-top-n", type=int, default=12)
    parser.add_argument("--minimum-reranker-score", type=float, default=float("-inf"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    parser.add_argument("--with-shuffled-control", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.retrieval)
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if args.label_order:
        names = [name.strip().lower() for name in args.label_order.split(",")]
        if len(names) != int(model.config.num_labels):
            raise ValueError(
                f"label order has {len(names)} names for {model.config.num_labels} logits"
            )
        model.config.id2label = {index: name for index, name in enumerate(names)}
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        args.reranker,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    reranker_model = reranker_model.to(
        "cuda" if torch.cuda.is_available() else "cpu"
    ).eval()
    conditions = [False, True] if args.with_shuffled_control else [False]
    output = []
    for shuffled in conditions:
        output.extend(run_condition(
            rows,
            model=model,
            tokenizer=tokenizer,
            reranker_model=reranker_model,
            reranker_tokenizer=reranker_tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_candidates=args.max_candidates,
            rerank_top_n=args.rerank_top_n,
            minimum_reranker_score=args.minimum_reranker_score,
            minimum_confidence=args.minimum_confidence,
            top_k=args.top_k,
            shuffle_control=shuffled,
            seed=args.seed,
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )


if __name__ == "__main__":
    main()
