#!/usr/bin/env python3
"""Train a lightweight black-box text probe on the public dev splits.

The protocol is intentionally strict:
1. fit each candidate on ``train`` only,
2. choose representation / regularization / threshold on ``validation``,
3. evaluate the selected candidate once on ``test`` when requested.

This is a trained black-box component: it uses only the transcript text and the
provided labels, not activations or weights of the evaluated model.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

LEADERBOARD_SRC = ROOT / "leaderboard" / "src"
sys.path.insert(0, str(LEADERBOARD_SRC))

from aletheia_runner.config import DatasetConfig  # noqa: E402


HF_SPLIT = "test"


@dataclasses.dataclass(frozen=True)
class Candidate:
    view: str
    features: str
    c: float
    class_weight: str | None
    max_features: int


@dataclasses.dataclass
class SplitData:
    frame: pd.DataFrame
    datasets: list[str]


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


def final_assistant(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


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


def text_views(messages: list[dict[str, Any]], max_context_chars: int) -> dict[str, str]:
    merged = merge_messages(messages)
    output = final_assistant(merged)
    dialogue = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in merged
    )
    if len(dialogue) > max_context_chars:
        dialogue = dialogue[-max_context_chars:]
    return {
        "output": output,
        "dialogue": dialogue,
        "output_context": f"CONTEXT:\n{dialogue}\n\nFINAL_ASSISTANT:\n{output}",
    }


def load_examples_for_labels(
    dataset_name: str,
    labels: pd.DataFrame,
    *,
    max_context_chars: int,
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
        views = text_views(row["messages"], max_context_chars)
        rows.append({
            "dataset": dataset_name,
            "index": index,
            "label": int(label_by_index[index]),
            **views,
        })

    order = {idx: i for i, idx in enumerate(labels["index"].tolist())}
    rows.sort(key=lambda item: order[item["index"]])
    if len(rows) != len(labels):
        raise RuntimeError(
            f"{dataset_name}: loaded {len(rows)} examples for {len(labels)} labels"
        )
    return pd.DataFrame(rows)


def load_split(split: str, splits_dir: Path, *, max_context_chars: int) -> SplitData:
    configs = load_split_config(splits_dir / f"dry.{split}.yaml", ROOT)
    frames = []
    for cfg in configs:
        labels = load_labels(cfg)
        frames.append(
            load_examples_for_labels(
                cfg.name,
                labels,
                max_context_chars=max_context_chars,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    return SplitData(frame=frame, datasets=[cfg.name for cfg in configs])


def build_pipeline(candidate: Candidate) -> Pipeline:
    transformers = []
    if "word" in candidate.features:
        transformers.append((
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=candidate.max_features,
                token_pattern=r"(?u)\b\w+\b",
            ),
        ))
    if "char" in candidate.features:
        transformers.append((
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=candidate.max_features,
            ),
        ))
    if not transformers:
        raise ValueError(f"no vectorizers for {candidate.features!r}")

    return Pipeline([
        ("features", FeatureUnion(transformers)),
        ("clf", LogisticRegression(
            C=candidate.c,
            class_weight=candidate.class_weight,
            max_iter=2000,
            solver="liblinear",
            random_state=0,
        )),
    ])


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
    per_dataset = []
    for _, group in frame.groupby("dataset", sort=True):
        per_dataset.append(binary_metrics(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            threshold,
        ))
    out = {}
    for key in ["balanced_accuracy", "auroc", "recall", "fpr"]:
        values = [m[key] for m in per_dataset if m[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def macro_threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | None]:
    per_dataset = []
    for _, group in frame.groupby("dataset", sort=True):
        per_dataset.append(threshold_metrics(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            threshold,
        ))
    out = {}
    for key in ["balanced_accuracy", "recall", "fpr"]:
        values = [m[key] for m in per_dataset if m[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def select_threshold(frame: pd.DataFrame) -> tuple[float, dict[str, float | None]]:
    best_threshold = 0.5
    best_metrics = macro_threshold_metrics(frame, best_threshold)
    for threshold in np.linspace(0.0, 1.0, 401):
        metrics = macro_threshold_metrics(frame, float(threshold))
        current = (
            metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0,
            -(metrics["fpr"] if metrics["fpr"] is not None else 1.0),
        )
        best = (
            best_metrics["balanced_accuracy"] if best_metrics["balanced_accuracy"] is not None else -1.0,
            -(best_metrics["fpr"] if best_metrics["fpr"] is not None else 1.0),
        )
        if current > best:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, macro_metrics(frame, best_threshold)


def add_scores(model: Pipeline, data: SplitData, view: str) -> pd.DataFrame:
    scores = model.predict_proba(data.frame[view].tolist())[:, 1]
    out = data.frame[["dataset", "index", "label"]].copy()
    out["score"] = scores
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


def candidate_grid(preset: str) -> list[Candidate]:
    views = ["output", "dialogue", "output_context"]
    features = ["word", "char", "word_char"]
    if preset == "quick":
        cs = [0.1, 1.0, 3.0]
        class_weights = ["balanced"]
        max_features = [50_000]
    elif preset == "full":
        cs = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        class_weights = ["balanced", None]
        max_features = [25_000, 75_000]
    else:
        raise ValueError(f"unknown preset={preset!r}")
    return [
        Candidate(view=view, features=feature, c=c, class_weight=class_weight, max_features=max_feature)
        for view in views
        for feature in features
        for c in cs
        for class_weight in class_weights
        for max_feature in max_features
    ]


def write_predictions(path: Path, frame: pd.DataFrame, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "index": frame["index"],
        "deceptive": frame["score"] >= threshold,
        "score": frame["score"],
        "label": frame["label"],
        "dataset": frame["dataset"],
    }).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "dev_splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "blackbox" / "text_probe_ngram_v1")
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading train split")
    train = load_split("train", args.splits_dir, max_context_chars=args.max_context_chars)
    print(f"train rows={len(train.frame)} datasets={len(train.datasets)} positives={int(train.frame['label'].sum())}")
    print("loading validation split")
    validation = load_split("validation", args.splits_dir, max_context_chars=args.max_context_chars)
    print(f"validation rows={len(validation.frame)} datasets={len(validation.datasets)} positives={int(validation.frame['label'].sum())}")

    grid_rows = []
    best: dict[str, Any] | None = None
    candidates = candidate_grid(args.preset)
    for i, candidate in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] {candidate}")
        model = build_pipeline(candidate)
        model.fit(train.frame[candidate.view].tolist(), train.frame["label"].to_numpy())
        validation_scores = add_scores(model, validation, candidate.view)
        threshold, metrics = select_threshold(validation_scores)
        row = {
            **dataclasses.asdict(candidate),
            "threshold": threshold,
            **metrics,
        }
        grid_rows.append(row)
        if best is None:
            is_best = True
        else:
            current = (
                row["balanced_accuracy"] if row["balanced_accuracy"] is not None else -1.0,
                row["auroc"] if row["auroc"] is not None else -1.0,
                -(row["fpr"] if row["fpr"] is not None else 1.0),
            )
            previous = (
                best["validation"]["metrics"]["balanced_accuracy"]
                if best["validation"]["metrics"]["balanced_accuracy"] is not None else -1.0,
                best["validation"]["metrics"]["auroc"]
                if best["validation"]["metrics"]["auroc"] is not None else -1.0,
                -(best["validation"]["metrics"]["fpr"]
                  if best["validation"]["metrics"]["fpr"] is not None else 1.0),
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
        raise RuntimeError("no candidate was evaluated")

    pd.DataFrame(grid_rows).sort_values(
        ["balanced_accuracy", "auroc", "fpr"],
        ascending=[False, False, True],
    ).to_csv(output_dir / "validation_grid.csv", index=False)

    threshold = float(best["threshold"])
    selected_model = best["model"]
    selected = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "fit train, select hyperparameters and threshold on validation",
        "candidate": best["candidate"],
        "threshold": threshold,
        "train": {
            "rows": int(len(train.frame)),
            "datasets": train.datasets,
            "positives": int(train.frame["label"].sum()),
        },
        "validation": best["validation"],
    }
    write_predictions(output_dir / "predictions" / "validation.csv", best["validation_scores"], threshold)

    if args.include_test:
        print("loading test split")
        test = load_split("test", args.splits_dir, max_context_chars=args.max_context_chars)
        test_scores = add_scores(selected_model, test, best["candidate"]["view"])
        selected["test"] = {
            "rows": int(len(test.frame)),
            "datasets": test.datasets,
            "positives": int(test.frame["label"].sum()),
            "metrics": macro_metrics(test_scores, threshold),
            "datasets_table": per_dataset_table(test_scores, threshold),
        }
        write_predictions(output_dir / "predictions" / "test.csv", test_scores, threshold)

    result_for_json = {k: v for k, v in selected.items() if k != "model"}
    (output_dir / "result.json").write_text(json.dumps(result_for_json, indent=2) + "\n")
    joblib.dump({
        "model": selected_model,
        "candidate": best["candidate"],
        "threshold": threshold,
        "max_context_chars": args.max_context_chars,
    }, output_dir / "model.joblib")

    print(json.dumps(result_for_json, indent=2))


if __name__ == "__main__":
    main()
