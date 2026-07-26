#!/usr/bin/env python3
"""Compare per-combo AUROC with AUROC pooled over a whole dev dataset.

The leaderboard reports one AUROC per *dataset* (Metis, Eunomia, Notus, Iris),
not one per (dataset, model, lora) combination.  Pooling mixes rows from
different base models and different organisms into a single ranking, so a
score that is only comparable *within* one organism loses most of its value.
This script measures that loss on the dev units.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/nonlinear_probe"))
from token_probes import TokenProbe, build_token_probe_model

LAYER = 46
PROBE_DIR = {"gemma": "gemma_probe", "qwen": "qwen_probe", "nemotron": "nemotron_probe"}
HIDDEN_DIM = {"gemma": 5376, "qwen": 5120, "nemotron": 4096}


def load_probes() -> dict:
    """Load the three shipped probes exactly as the submission notebook does."""
    probes = {}
    for family in ("gemma", "qwen", "nemotron"):
        directory = REPO / "submission/whitebox_probe" / PROBE_DIR[family]
        config = json.load(open(directory / "config.json"))
        network = build_token_probe_model(config["architecture"], HIDDEN_DIM[family])
        network.load_state_dict(torch.load(directory / "model.pt", map_location="cpu"))
        network.eval()
        probe = TokenProbe(config["architecture"], seed=0, device="cpu",
                           batch_token_budget=8192)
        probe.model = network
        probe.feature_mean = torch.load(directory / "feature_mean.pt", map_location="cpu")
        probe.feature_std = torch.load(directory / "feature_std.pt", map_location="cpu")
        probes[family] = probe
    return probes


def judge_margins() -> dict:
    """Return ``(dataset, index) -> logit_1 - logit_0`` from the local cache."""
    cache = json.load(open(REPO / "experiments/ensemble_gate_eval/direct_judge_cache.json"))
    margins = {}
    for key, (logit_0, logit_1) in cache.items():
        dataset, index = key.split("|", 1)
        margins[(dataset, index)] = float(logit_1) - float(logit_0)
    return margins


def score_dataset(dataset: str, rows: list, probes: dict) -> tuple:
    """Return ``(labels, probe_logit, judge_margin, family)`` for one dev unit."""
    family = ("nemotron" if "Nemotron" in dataset
              else "gemma" if "gemma" in dataset else "qwen")
    cache_file = REPO / f"results/whitebox/activations/{dataset.replace('/', '__')}.tokens.npz"
    data = dict(np.load(cache_file, allow_pickle=True))
    tokens_key = [key for key in data if key.startswith(f"tokens_L{LAYER}")][0]
    all_tokens = data[tokens_key]
    offsets = data["token_offsets"].astype(np.int64)
    position = {str(value): row for row, value in enumerate(data["index"])}

    kept = [row for row in rows if row[0] in position]
    row_ids = [position[row[0]] for row in kept]
    pieces = [all_tokens[offsets[r]:offsets[r + 1]] for r in row_ids]
    flat = torch.from_numpy(np.concatenate(pieces, axis=0)).float()
    new_offsets = np.cumsum([0] + [len(piece) for piece in pieces]).astype(np.int64)

    logits = np.nan_to_num(probes[family].decision_function(flat, new_offsets), nan=0.0)
    labels = np.array([row[1] for row in kept])
    margins = np.array([row[2] for row in kept])
    return labels, logits, margins, family


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, or NaN when the rows are single-class."""
    return roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")


def main() -> None:
    constants = json.load(open(REPO / "results/ensemble_gate_eval/sonic_v3_constants.json"))
    lam = constants["lambda_blend"]
    judge_sd = constants["judge_margin_sd"]
    probe_sd = constants["probe_logit_sd"]
    probes = load_probes()
    margins = judge_margins()

    by_dataset = {}
    for row in csv.DictReader(open(REPO / "dev_splits/manifest.csv")):
        key = (row["dataset"], str(row["index"]))
        if key not in margins:
            continue
        by_dataset.setdefault(row["dataset"], []).append(
            (str(row["index"]), row["deceptive"] == "True", margins[key]))

    pools = {"instructed": [], "varied": []}
    per_combo = {"instructed": [], "varied": []}
    print(f"{'dataset':<48}{'probe':>8}{'judge':>8}{'blend':>8}{'p.mean':>9}{'p.sd':>7}")
    for dataset in sorted(by_dataset):
        labels, logits, margin, family = score_dataset(dataset, by_dataset[dataset], probes)
        z_probe = logits / probe_sd[family]
        z_judge = margin / judge_sd
        blend = (1 - lam) * z_probe + lam * z_judge
        pool = "instructed" if "instructed" in dataset else "varied"
        pools[pool].append((labels, z_probe, z_judge, blend))
        per_combo[pool].append((auroc(labels, z_probe), auroc(labels, z_judge),
                                auroc(labels, blend)))
        print(f"{dataset.split('-deception-')[-1][:47]:<48}"
              f"{auroc(labels, z_probe):>8.4f}{auroc(labels, z_judge):>8.4f}"
              f"{auroc(labels, blend):>8.4f}{z_probe.mean():>9.3f}{z_probe.std():>7.3f}")

    print(f"\n{'pool':<14}{'n':>6}{'':>4}{'probe':>9}{'judge':>9}{'blend':>9}")
    for pool, entries in pools.items():
        labels = np.concatenate([entry[0] for entry in entries])
        mean_combo = np.nanmean(np.array(per_combo[pool]), axis=0)
        pooled = [auroc(labels, np.concatenate([entry[column] for entry in entries]))
                  for column in (1, 2, 3)]
        print(f"{pool + ' (mean)':<14}{len(labels):>6}{'':>4}"
              + "".join(f"{value:>9.4f}" for value in mean_combo))
        print(f"{pool + ' (pooled)':<14}{len(labels):>6}{'':>4}"
              + "".join(f"{value:>9.4f}" for value in pooled))


if __name__ == "__main__":
    main()
