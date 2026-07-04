#!/usr/bin/env python3
"""Train a BGE embedding probe on the public dev splits.

This is a standalone black-box text-feature experiment. It embeds transcript
views with ``BAAI/bge-base-en-v1.5`` using normalized CLS embeddings, trains a
lightweight classifier on train, selects hyperparameters and threshold on
validation, then optionally reports test once.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.text_probe.run_text_probe import (  # noqa: E402
    SplitData,
    binary_metrics,
    load_split,
    macro_metrics,
    per_dataset_table,
    select_threshold,
    write_predictions,
)


MODEL_ID = "BAAI/bge-base-en-v1.5"


@dataclasses.dataclass(frozen=True)
class Candidate:
    view: str
    classifier: str
    c: float | None = None
    hidden: int | None = None
    alpha: float | None = None


def text_column(data: SplitData, view: str) -> list[str]:
    return data.frame[view].fillna("").astype(str).tolist()


def cache_key(model_id: str, view: str, split: str, max_length: int, quantize: str) -> str:
    safe_model = model_id.replace("/", "__")
    return f"{safe_model}.{view}.{split}.len{max_length}.{quantize}.npy"


def load_encoder(model_id: str, *, quantize: str) -> tuple[Any, torch.nn.Module]:
    # Avoid a very slow importlib.metadata scan over the cluster's system Python
    # packages during transformers import. The distribution mapping is not needed
    # for this direct AutoTokenizer/AutoModel use.
    importlib.metadata.packages_distributions = lambda: {}
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    if quantize == "dynamic_int8":
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    elif quantize != "none":
        raise ValueError(f"unknown quantize={quantize!r}")
    return tokenizer, model


def encode_texts(
    texts: list[str],
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    batch_size: int,
    max_length: int,
    progress_label: str,
) -> np.ndarray:
    chunks = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            output = model(**encoded)
            # BGE's model card recommends CLS pooling for transformers usage.
            embeddings = output.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            chunks.append(embeddings.cpu().numpy().astype("float32"))
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                print(f"  {progress_label} batch {batch_index}/{total_batches}", flush=True)
    return np.concatenate(chunks, axis=0)


def get_embeddings(
    data: SplitData,
    *,
    split: str,
    view: str,
    tokenizer: Any,
    model: torch.nn.Module,
    cache_dir: Path,
    batch_size: int,
    max_length: int,
    quantize: str,
    refresh_cache: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / cache_key(MODEL_ID, view, split, max_length, quantize)
    if path.exists() and not refresh_cache:
        arr = np.load(path)
        if arr.shape[0] != len(data.frame):
            raise RuntimeError(f"{path} has {arr.shape[0]} rows, expected {len(data.frame)}")
        return arr
    print(f"encoding split={split} view={view} rows={len(data.frame)}")
    arr = encode_texts(
        text_column(data, view),
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        progress_label=f"{split}/{view}",
    )
    np.save(path, arr)
    return arr


def candidate_grid(preset: str, views: list[str]) -> list[Candidate]:
    if preset == "quick":
        return [
            Candidate(view=view, classifier="logreg", c=c)
            for view in views
            for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        ]
    if preset == "mlp":
        return [
            *[
                Candidate(view=view, classifier="logreg", c=c)
                for view in views
                for c in [0.1, 1.0, 10.0]
            ],
            *[
                Candidate(view=view, classifier="mlp", hidden=hidden, alpha=alpha)
                for view in views
                for hidden in [64, 128]
                for alpha in [0.0001, 0.001, 0.01]
            ],
        ]
    raise ValueError(f"unknown preset={preset!r}")


def build_classifier(candidate: Candidate) -> Pipeline:
    if candidate.classifier == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=float(candidate.c),
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=0,
            )),
        ])
    if candidate.classifier == "mlp":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(int(candidate.hidden),),
                alpha=float(candidate.alpha),
                batch_size=128,
                early_stopping=True,
                max_iter=400,
                random_state=0,
            )),
        ])
    raise ValueError(f"unknown classifier={candidate.classifier!r}")


def score_frame(model: Pipeline, data: SplitData, embeddings: np.ndarray) -> Any:
    scores = model.predict_proba(embeddings)[:, 1]
    out = data.frame[["dataset", "index", "label"]].copy()
    out["score"] = scores
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "blackbox" / "bge_probe_v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "blackbox" / "bge_probe_v1" / "embeddings")
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--views", default="output_context",
                        help="comma-separated subset of output,dialogue,output_context")
    parser.add_argument("--preset", choices=["quick", "mlp"], default="quick")
    parser.add_argument("--quantize", choices=["dynamic_int8", "none"], default="dynamic_int8")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    invalid = sorted(set(views) - {"output", "dialogue", "output_context"})
    if invalid:
        raise SystemExit(f"invalid views: {invalid}")
    print(f"torch_threads={torch.get_num_threads()} views={views} max_length={args.max_length}")

    print("loading train split")
    train = load_split("train", args.splits_dir, max_context_chars=args.max_context_chars)
    print(f"train rows={len(train.frame)} datasets={len(train.datasets)} positives={int(train.frame['label'].sum())}")
    print("loading validation split")
    validation = load_split("validation", args.splits_dir, max_context_chars=args.max_context_chars)
    print(f"validation rows={len(validation.frame)} datasets={len(validation.datasets)} positives={int(validation.frame['label'].sum())}")

    print(f"loading encoder model={MODEL_ID} quantize={args.quantize}")
    tokenizer, encoder = load_encoder(MODEL_ID, quantize=args.quantize)

    grid_rows = []
    best: dict[str, Any] | None = None
    embedding_cache: dict[tuple[str, str], np.ndarray] = {}

    def embeddings_for(data: SplitData, split: str, view: str) -> np.ndarray:
        key = (split, view)
        if key not in embedding_cache:
            embedding_cache[key] = get_embeddings(
                data,
                split=split,
                view=view,
                tokenizer=tokenizer,
                model=encoder,
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
                max_length=args.max_length,
                quantize=args.quantize,
                refresh_cache=args.refresh_cache,
            )
        return embedding_cache[key]

    candidates = candidate_grid(args.preset, views)
    for i, candidate in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] {candidate}")
        x_train = embeddings_for(train, "train", candidate.view)
        x_validation = embeddings_for(validation, "validation", candidate.view)
        model = build_classifier(candidate)
        model.fit(x_train, train.frame["label"].to_numpy())
        validation_scores = score_frame(model, validation, x_validation)
        threshold, metrics = select_threshold(validation_scores)
        row = {
            **dataclasses.asdict(candidate),
            "threshold": threshold,
            **metrics,
        }
        grid_rows.append(row)
        current = (
            metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0,
            metrics["auroc"] if metrics["auroc"] is not None else -1.0,
            -(metrics["fpr"] if metrics["fpr"] is not None else 1.0),
        )
        if best is None:
            is_best = True
        else:
            previous_metrics = best["validation"]["metrics"]
            previous = (
                previous_metrics["balanced_accuracy"] if previous_metrics["balanced_accuracy"] is not None else -1.0,
                previous_metrics["auroc"] if previous_metrics["auroc"] is not None else -1.0,
                -(previous_metrics["fpr"] if previous_metrics["fpr"] is not None else 1.0),
            )
            is_best = current > previous
        if is_best:
            best = {
                "candidate": dataclasses.asdict(candidate),
                "threshold": threshold,
                "model": model,
                "validation_scores": validation_scores,
                "validation": {
                    "metrics": metrics,
                    "datasets": per_dataset_table(validation_scores, threshold),
                },
            }

    if best is None:
        raise RuntimeError("no candidates evaluated")

    import pandas as pd

    pd.DataFrame(grid_rows).sort_values(
        ["balanced_accuracy", "auroc", "fpr"],
        ascending=[False, False, True],
    ).to_csv(args.output_dir / "validation_grid.csv", index=False)

    threshold = float(best["threshold"])
    write_predictions(args.output_dir / "predictions" / "validation.csv", best["validation_scores"], threshold)
    selected = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "BGE embeddings; fit train, select hyperparameters and threshold on validation",
        "model_id": MODEL_ID,
        "quantize": args.quantize,
        "max_length": args.max_length,
        "candidate": best["candidate"],
        "threshold": threshold,
        "train": {
            "rows": int(len(train.frame)),
            "datasets": train.datasets,
            "positives": int(train.frame["label"].sum()),
        },
        "validation": best["validation"],
    }

    if args.include_test:
        print("loading test split")
        test = load_split("test", args.splits_dir, max_context_chars=args.max_context_chars)
        x_test = embeddings_for(test, "test", best["candidate"]["view"])
        test_scores = score_frame(best["model"], test, x_test)
        selected["test"] = {
            "rows": int(len(test.frame)),
            "datasets": test.datasets,
            "positives": int(test.frame["label"].sum()),
            "metrics": macro_metrics(test_scores, threshold),
            "datasets_table": per_dataset_table(test_scores, threshold),
        }
        write_predictions(args.output_dir / "predictions" / "test.csv", test_scores, threshold)

    result_for_json = {k: v for k, v in selected.items() if k != "model"}
    (args.output_dir / "result.json").write_text(json.dumps(result_for_json, indent=2) + "\n")
    joblib.dump({
        "classifier": best["model"],
        "candidate": best["candidate"],
        "threshold": threshold,
        "model_id": MODEL_ID,
        "quantize": args.quantize,
        "max_length": args.max_length,
    }, args.output_dir / "classifier.joblib")

    print(json.dumps(result_for_json, indent=2))


if __name__ == "__main__":
    main()
