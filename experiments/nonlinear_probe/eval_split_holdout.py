"""Leakage-free pre-submission eval of the token probe on the frozen dev splits.

For each base model the probe is refit on the ``train`` rows of ``dev_splits``
only (pooled across that model's datasets, same recipe as the deployed probe:
transformer token probe at layer 46, seed 0, 60 epochs) and scored on the
held-out ``test`` rows. The exported all-rows submission weights are scored on
the same test rows, so the gap quantifies how much training on every row
(including the test rows) inflated the earlier numbers.

Reads cached activations from ``results/whitebox/activations`` -- no NDIF.

    python experiments/nonlinear_probe/eval_split_holdout.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_probes import TokenProbe, build_token_probe_model  # noqa: E402

ACTIVATIONS = REPO_ROOT / "results/whitebox/activations"
MANIFEST = REPO_ROOT / "dev_splits/manifest.csv"
SUBMISSION_PROBES = REPO_ROOT / "submission/whitebox_probe"
OUT_JSON = REPO_ROOT / "results/whitebox/split_holdout_eval.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
MAX_EPOCHS = 60
PROBE_DIR = {"gemma": "gemma_probe", "qwen": "qwen_probe", "nemotron": "nemotron_probe"}


def base_model(dataset_name: str) -> str:
    """Map a dataset name to the base model whose probe covers it."""
    if "gemma-3-27b-it" in dataset_name:
        return "gemma"
    if "Nemotron" in dataset_name:
        return "nemotron"
    if "Qwen3.5-27B" in dataset_name:
        return "qwen"
    raise ValueError(f"unknown base model for {dataset_name}")


def load_split_map() -> dict[tuple[str, str], str]:
    """(dataset, str(index)) -> dev_split from the frozen manifest."""
    split_map: dict[tuple[str, str], str] = {}
    with open(MANIFEST) as handle:
        for row in csv.DictReader(handle):
            split_map[(row["dataset"], str(row["index"]))] = row["dev_split"]
    return split_map


def cache_path(dataset_name: str) -> Path:
    return ACTIVATIONS / f"{dataset_name.replace('/', '__')}.tokens.npz"


def datasets_for_model(split_map: dict, model: str) -> list[str]:
    names = {dataset for dataset, _ in split_map if base_model(dataset) == model}
    return sorted(names)


def pool_split(model: str, split_map: dict, layer: int, wanted_split: str):
    """Concatenate one split's token spans across a model's datasets.

    :return: (flat float16 tensor on DEVICE, offsets int64, labels int array,
        per-row dataset name array).
    """
    spans: list[np.ndarray] = []
    labels: list[int] = []
    row_dataset: list[str] = []
    for dataset in datasets_for_model(split_map, model):
        with np.load(cache_path(dataset), allow_pickle=True) as data:
            tokens = data[f"tokens_L{layer}"]
            offsets = data["token_offsets"].astype(np.int64)
            indices = data["index"]
            deceptive = data["deceptive"].astype(int)
        for row in range(len(indices)):
            if split_map.get((dataset, str(indices[row]))) != wanted_split:
                continue
            spans.append(tokens[offsets[row]:offsets[row + 1]])
            labels.append(int(deceptive[row]))
            row_dataset.append(dataset)
    lengths = [len(span) for span in spans]
    flat = np.concatenate(spans, axis=0)
    new_offsets = np.cumsum([0] + lengths).astype(np.int64)
    finfo = torch.finfo(torch.float16)
    flat_tensor = torch.from_numpy(flat).clamp(finfo.min, finfo.max).to(DEVICE)
    return flat_tensor, new_offsets, np.asarray(labels), np.asarray(row_dataset, dtype=object)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Balanced-accuracy-optimal cut over the observed scores."""
    cuts = np.unique(scores)
    accuracies = [balanced_accuracy_score(labels, scores >= cut) for cut in cuts]
    return float(cuts[int(np.argmax(accuracies))])


def per_dataset_metrics(labels, scores, row_dataset, threshold):
    out = {}
    for dataset in sorted(set(row_dataset)):
        mask = row_dataset == dataset
        y = labels[mask]
        s = scores[mask]
        auroc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
        ba = balanced_accuracy_score(y, s >= threshold)
        out[dataset.split("/")[-1]] = {"n": int(mask.sum()), "auroc": auroc,
                                       "balanced_accuracy": ba, "deceptive": int(y.sum())}
    return out


def load_submission_probe(model: str, hidden_dim: int):
    """Rebuild the exported all-rows probe for scoring (the leaked baseline)."""
    probe_dir = SUBMISSION_PROBES / PROBE_DIR[model]
    config = json.loads((probe_dir / "config.json").read_text())
    net = build_token_probe_model(config["architecture"], hidden_dim).to(DEVICE)
    net.load_state_dict(torch.load(probe_dir / "model.pt", map_location=DEVICE))
    net.eval()
    probe = TokenProbe(config["architecture"], seed=SEED, device=DEVICE,
                       batch_token_budget=config.get("batch_token_budget", 8192))
    probe.model = net
    probe.feature_mean = torch.load(probe_dir / "feature_mean.pt", map_location=DEVICE)
    probe.feature_std = torch.load(probe_dir / "feature_std.pt", map_location=DEVICE)
    return probe, config


def mean_metric(per_dataset: dict, key: str) -> float:
    values = [v[key] for v in per_dataset.values() if not np.isnan(v[key])]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    split_map = load_split_map()
    report = {}
    for model in ("gemma", "nemotron", "qwen"):
        config_layer = json.loads(
            (SUBMISSION_PROBES / PROBE_DIR[model] / "config.json").read_text())["layer"]
        train_flat, train_off, train_y, _ = pool_split(model, split_map, config_layer, "train")
        test_flat, test_off, test_y, test_ds = pool_split(model, split_map, config_layer, "test")
        print(f"\n=== {model}  layer L{config_layer}  "
              f"train={len(train_y)} ({train_y.sum()} dec)  "
              f"test={len(test_y)} ({test_y.sum()} dec) ===", flush=True)

        # Honest probe: refit on train rows only, score held-out test rows.
        probe = TokenProbe("transformer", seed=SEED, device=DEVICE,
                           max_epochs=MAX_EPOCHS).fit(train_flat, train_off, train_y)
        train_scores = probe.predict_proba(train_flat, train_off)[:, 1]
        honest_thr = best_threshold(train_y, train_scores)
        honest_scores = probe.predict_proba(test_flat, test_off)[:, 1]
        honest = per_dataset_metrics(test_y, honest_scores, test_ds, honest_thr)

        # Leaked baseline: exported all-rows weights scored on the same test rows.
        leaked_probe, leaked_config = load_submission_probe(model, test_flat.shape[1])
        leaked_scores = leaked_probe.predict_proba(test_flat, test_off)[:, 1]
        leaked_thr = leaked_config.get("threshold", 0.5)
        leaked = per_dataset_metrics(test_y, leaked_scores, test_ds, leaked_thr)

        report[model] = {
            "layer": config_layer, "n_train": int(len(train_y)), "n_test": int(len(test_y)),
            "honest_threshold": honest_thr, "leaked_threshold": leaked_thr,
            "honest_mean_auroc": mean_metric(honest, "auroc"),
            "honest_mean_ba": mean_metric(honest, "balanced_accuracy"),
            "leaked_mean_auroc": mean_metric(leaked, "auroc"),
            "leaked_mean_ba": mean_metric(leaked, "balanced_accuracy"),
            "honest_per_dataset": honest, "leaked_per_dataset": leaked,
        }
        print(f"  honest (train-refit)  mean AUROC={report[model]['honest_mean_auroc']:.4f}  "
              f"mean BA={report[model]['honest_mean_ba']:.4f}")
        print(f"  leaked (all-rows)     mean AUROC={report[model]['leaked_mean_auroc']:.4f}  "
              f"mean BA={report[model]['leaked_mean_ba']:.4f}")
        for name in report[model]["honest_per_dataset"]:
            h = honest[name]
            l = leaked[name]
            print(f"    {name:<52} n={h['n']:<3} "
                  f"AUROC {h['auroc']:.3f}/{l['auroc']:.3f}  "
                  f"BA {h['balanced_accuracy']:.3f}/{l['balanced_accuracy']:.3f}  (honest/leaked)")

    # Overall means weighted equally across the 21 test datasets.
    all_honest = {f"{m}:{d}": v for m, r in report.items() for d, v in r["honest_per_dataset"].items()}
    all_leaked = {f"{m}:{d}": v for m, r in report.items() for d, v in r["leaked_per_dataset"].items()}
    report["overall"] = {
        "n_datasets": len(all_honest),
        "honest_mean_auroc": mean_metric(all_honest, "auroc"),
        "honest_mean_ba": mean_metric(all_honest, "balanced_accuracy"),
        "leaked_mean_auroc": mean_metric(all_leaked, "auroc"),
        "leaked_mean_ba": mean_metric(all_leaked, "balanced_accuracy"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    o = report["overall"]
    print(f"\n=== OVERALL over {o['n_datasets']} test datasets ===")
    print(f"  honest (leakage-free)  mean AUROC={o['honest_mean_auroc']:.4f}  "
          f"mean BA={o['honest_mean_ba']:.4f}")
    print(f"  leaked  (all-rows)     mean AUROC={o['leaked_mean_auroc']:.4f}  "
          f"mean BA={o['leaked_mean_ba']:.4f}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
