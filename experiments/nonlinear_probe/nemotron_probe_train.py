"""
Layer-select, train, and export the Nemotron transformer token probe.

The Nemotron dev split is one dataset (216 rows, 16 deceptive), so this is a
single-cache pipeline: stratified k-fold out-of-fold scores per candidate
layer, pick the best layer by OOF AUROC, refit on all rows with the same
recipe as the Qwen/gemma probes (seed 0, max_epochs 60), and export the probe
package to submission/whitebox_probe/nemotron_probe/.

Usage (from the repo root):

    python experiments/nonlinear_probe/nemotron_probe_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from token_probes import TokenProbe  # noqa: E402

CACHE = (REPO_ROOT / "results/whitebox/activations/"
         "aletheias-quest__dev-instructed-deception-"
         "NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None.tokens.npz")
OUT_DIR = REPO_ROOT / "submission/whitebox_probe/nemotron_probe"
N_FOLDS = 5

# Nemotron's NDIF deployment runs with very little GPU headroom and its
# Mamba-mixer memory grows with batch width, so the deployed notebook must
# trace one row at a time.
EXTRACT_TOKEN_BUDGET = 512
EXTRACT_MAX_BATCH = 1


def load_layer(cache: dict, layer: int, device: str) -> torch.Tensor:
    """
    :param cache: Open npz mapping holding tokens_L{layer} arrays.
    :param layer: Decoder layer index to load.
    :param device: Torch device for the returned tensor.
    :return: (total_tokens, hidden) float16 tensor of that layer's activations.
    """
    finfo = torch.finfo(torch.float16)
    tokens = torch.from_numpy(cache[f"tokens_L{layer}"])
    return tokens.clamp(finfo.min, finfo.max).to(device)


def out_of_fold_scores(flat: torch.Tensor, offsets: np.ndarray,
                       labels: np.ndarray, device: str) -> np.ndarray:
    """
    :param flat: (total_tokens, hidden) activations on device.
    :param offsets: (N+1,) token offsets per example.
    :param labels: (N,) binary labels.
    :param device: Torch device to train on.
    :return: (N,) out-of-fold deception probabilities.
    """
    scores = np.zeros(len(labels))
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for fold_index, (train_rows, test_rows) in enumerate(folds.split(scores, labels)):
        train_flat, train_offsets = subset(flat, offsets, train_rows)
        test_flat, test_offsets = subset(flat, offsets, test_rows)
        probe = TokenProbe("transformer", seed=fold_index, device=device,
                           max_epochs=60).fit(train_flat, train_offsets,
                                              labels[train_rows])
        scores[test_rows] = probe.predict_proba(test_flat, test_offsets)[:, 1]
    return scores


def subset(flat: torch.Tensor, offsets: np.ndarray,
           rows: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """
    :param flat: (total_tokens, hidden) activations.
    :param offsets: (N+1,) token offsets per example.
    :param rows: Example indices to keep.
    :return: (concatenated activations, rebuilt offsets) for just those rows.
    """
    pieces = [flat[int(offsets[row]):int(offsets[row + 1])] for row in rows]
    lengths = [len(piece) for piece in pieces]
    new_offsets = np.cumsum([0] + lengths).astype(np.int64)
    return torch.cat(pieces), new_offsets


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(CACHE, allow_pickle=True) as data:
        layers = sorted(int(key.split("L")[1]) for key in data.files
                        if key.startswith("tokens_L"))
        offsets = data["token_offsets"].astype(np.int64)
        labels = data["deceptive"].astype(int)
        cache = {key: data[key] for key in data.files if key.startswith("tokens_L")}
    print(f"{len(labels)} examples ({labels.sum()} deceptive), "
          f"candidate layers {layers}", flush=True)

    results = []
    for layer in layers:
        flat = load_layer(cache, layer, device)
        oof = out_of_fold_scores(flat, offsets, labels, device)
        auroc = roc_auc_score(labels, oof)
        accuracy_default = balanced_accuracy_score(labels, oof >= 0.5)
        candidates = np.unique(oof)
        accuracies = [balanced_accuracy_score(labels, oof >= cutoff)
                      for cutoff in candidates]
        best_cut = int(np.argmax(accuracies))
        results.append((layer, auroc, accuracies[best_cut],
                        float(candidates[best_cut])))
        print(f"L{layer}: OOF AUROC={auroc:.4f}  BA@0.5={accuracy_default:.4f}  "
              f"BA@opt={accuracies[best_cut]:.4f} (thr={candidates[best_cut]:.3f})",
              flush=True)
        del flat
        torch.cuda.empty_cache()

    best_layer, best_auroc, best_accuracy, best_threshold = max(
        results, key=lambda row: (row[2], row[1]))
    print(f"selected L{best_layer} (BA@opt={best_accuracy:.4f}, "
          f"AUROC={best_auroc:.4f}, threshold={best_threshold:.3f})")

    flat = load_layer(cache, best_layer, device)
    probe = TokenProbe("transformer", seed=0, device=device,
                       max_epochs=60).fit(flat, offsets, labels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(probe.model.state_dict(), OUT_DIR / "model.pt")
    torch.save(probe.feature_mean, OUT_DIR / "feature_mean.pt")
    torch.save(probe.feature_std, OUT_DIR / "feature_std.pt")
    config = dict(architecture="transformer", layer=best_layer,
                  hidden_dim=flat.shape[1], d_model=128, n_heads=4,
                  dim_feedforward=256, n_blocks=2, dropout=0.1,
                  batch_token_budget=8192, threshold=round(best_threshold, 4),
                  extract_token_budget=EXTRACT_TOKEN_BUDGET,
                  extract_max_batch=EXTRACT_MAX_BATCH)
    with open(OUT_DIR / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)
    print(f"exported to {OUT_DIR}")


if __name__ == "__main__":
    main()
