"""
Standardized-split probe evaluation using dev_splits/manifest.csv.

Instead of pooling all rows of a scenario together and evaluating on the
other scenario (the original cross-scenario protocol), this script uses
deterministic 80/10/10 train/validation/test splits per dataset (created
by ``scripts/make_dev_splits.py``) so every probe sees the exact same test
rows. Multiple training-data compositions are compared:

- ``both``          train split, all scenarios
- ``varied``        train split, varied-deception only
- ``instructed``    train split, instructed-deception only
- ``both+val``      train + validation splits, all scenarios

Evaluation is always on the test split (all scenarios). Probes compared at
L46 (the peak from Phase 1-2 sweeps):

- logistic (concat mean+last pooled)
- mlp-512 (concat mean+last pooled)
- transformer token
- cnn token
- attention token

Usage:

    python experiments/nonlinear_probe/standardized_sweep.py

Output: results/whitebox/standardized_sweep/results.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from linear_sweep import (
    base_model_family,
    compute_metrics,
    fit_logistic,
    has_both_classes,
    is_limit_cache,
    is_tokens_cache,
    meta_model_id,
    parse_scenario,
    per_dataset_metrics,
    rows_to_frame,
)
from nonlinear_sweep import probe_label
from token_probes import TokenProbe

MANIFEST_PATH = Path("dev_splits/manifest.csv")
CACHE_DIR_DEFAULT = "results/whitebox/activations"
OUT_DIR_DEFAULT = "results/whitebox/standardized_sweep"
LAYER = 46
POOLING = "concat"

RESULT_COLUMNS = [
    "base_model", "probe", "train_data", "auroc",
    "balanced_accuracy", "n_train", "n_eval",
]
PER_DATASET_COLUMNS = [
    "base_model", "probe", "train_data", "dataset",
    "auroc", "balanced_accuracy", "n_eval",
]


def load_manifest(path: Path) -> pd.DataFrame:
    """Load the dev-split manifest and ensure index is int."""
    df = pd.read_csv(path)
    df = df.astype({"index": int})
    return df


def build_manifest_lookup(manifest: pd.DataFrame) -> dict:
    """Return dict: (dataset_name, index) -> {dev_split, scenario, deceptive}."""
    lookup = {}
    for _, row in manifest.iterrows():
        lookup[(row["dataset"], row["index"])] = {
            "dev_split": row["dev_split"],
            "scenario": row["scenario"],
            "deceptive": row["deceptive"],
        }
    return lookup


def discover_standardized_caches(
        cache_dir: Path, manifest: pd.DataFrame,
) -> list[dict]:
    """
    Find every cache (pooled and token) that has rows in the manifest,
    and for each cache determine which rows fall into each split.

    Returns a list of dicts, one per cache, with keys:
        path, dataset, base_model, model_id, scenario, layers,
        is_token, train_mask, val_mask, test_mask (bool arrays),
        scenario_of_row (object array), labels (int64), valid (bool).
    """
    manifest_lookup = build_manifest_lookup(manifest)
    datasets_in_manifest = set(manifest["dataset"].unique())

    caches: list[dict] = []
    for cache_path in sorted(cache_dir.glob("*.npz")):
        if is_limit_cache(cache_path.name):
            continue
        with np.load(cache_path, allow_pickle=True) as data:
            if "deceptive" not in data or "index" not in data:
                continue
            meta = json.loads(str(data["meta"]))
            dataset = meta["dataset"]
            if dataset not in datasets_in_manifest:
                continue

            indices = np.asarray(data["index"])
            raw_labels = np.asarray(data["deceptive"])
            # Bool labels are all valid; int labels use -1 for missing.
            if np.issubdtype(raw_labels.dtype, np.bool_):
                valid = np.ones(len(indices), dtype=bool)
            else:
                valid = raw_labels >= 0
            if not valid.any():
                print(f"  skipping {cache_path.name}: no usable labels")
                continue

            is_token = is_tokens_cache(cache_path.name)
            n_total = len(indices)
            train_mask = np.zeros(n_total, dtype=bool)
            val_mask = np.zeros(n_total, dtype=bool)
            test_mask = np.zeros(n_total, dtype=bool)
            scenario_of_row = np.full(n_total, "", dtype=object)

            for row_idx, idx in enumerate(indices):
                key = (dataset, int(idx))
                entry = manifest_lookup.get(key)
                if entry is None:
                    continue
                split = entry["dev_split"]
                if split == "train":
                    train_mask[row_idx] = True
                elif split == "validation":
                    val_mask[row_idx] = True
                elif split == "test":
                    test_mask[row_idx] = True
                scenario_of_row[row_idx] = entry["scenario"]

            if not test_mask.any():
                continue

        scenario = parse_scenario(dataset)
        model_id = meta_model_id(meta)
        caches.append(dict(
            path=cache_path,
            dataset=dataset,
            base_model=base_model_family(model_id),
            model_id=model_id,
            scenario=scenario,
            layers=list(meta["layers"]),
            is_token=is_token,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            scenario_of_row=scenario_of_row,
            labels=raw_labels[valid].astype(np.int64),
            valid=valid,
        ))
    return caches


def fit_mlp_local(features, labels, hidden_layers, alpha, seed):
    """Local copy of fit_mlp to avoid importing sklearn at module level."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=hidden_layers, alpha=alpha, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=10, max_iter=300,
            random_state=seed)),
    ])
    pipeline.fit(features, labels)
    return pipeline


def _pooled_rows(cache_info: dict, layer: int, pooling: str,
                 row_mask: np.ndarray) -> np.ndarray:
    """Read pooled features for selected rows from one cache."""
    keys = [f"mean_L{layer}", f"last_L{layer}"] if pooling == "concat"         else [f"{pooling}_L{layer}"]
    float16_max = float(np.finfo(np.float16).max)
    with np.load(cache_info["path"], allow_pickle=True) as data:
        parts = [
            np.nan_to_num(np.asarray(data[key])[row_mask].astype(np.float32),
                          nan=0.0, posinf=float16_max, neginf=-float16_max)
            for key in keys
        ]
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)


def _token_rows(cache_info: dict, layer: int, device: str,
                row_mask: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Read token features for selected rows from one cache.

    Returns (flat_features, offsets) where offsets are local to this cache
    (starting at 0).
    """
    with np.load(cache_info["path"], allow_pickle=True) as data:
        all_offsets = np.asarray(data["token_offsets"], dtype=np.int64)
        all_features = torch.from_numpy(
            np.asarray(data[f"tokens_L{layer}"])).to(device)

    # Clamp infinities (float16 overflow artifacts).
    finfo = torch.finfo(all_features.dtype)
    all_features = all_features.clamp(finfo.min, finfo.max)

    row_indices = np.where(row_mask)[0]
    n_rows = len(row_indices)
    if n_rows == 0:
        return torch.empty(0, all_features.shape[1], device=device),             np.array([0], dtype=np.int64)

    selected_starts = all_offsets[row_indices]
    selected_ends = all_offsets[row_indices + 1]
    selected_lengths = selected_ends - selected_starts

    # Build flat tensor of selected tokens.
    token_indices = []
    for s, e in zip(selected_starts, selected_ends):
        token_indices.extend(range(int(s), int(e)))
    selected_flat = all_features[token_indices]

    # Build offsets starting at 0.
    offsets = np.concatenate([[0], np.cumsum(selected_lengths)]).astype(np.int64)
    return selected_flat, offsets


def mask_fn_for_composition(composition: str):
    """Return a function that yields a train mask from a cache_info dict."""

    def _mask(cache_info: dict) -> np.ndarray:
        if composition == "both":
            return cache_info["train_mask"]
        elif composition == "varied":
            return cache_info["train_mask"] & (
                cache_info["scenario_of_row"] == "varied-deception")
        elif composition == "instructed":
            return cache_info["train_mask"] & (
                cache_info["scenario_of_row"] == "instructed-deception")
        elif composition == "both+val":
            return cache_info["train_mask"] | cache_info["val_mask"]
        else:
            raise ValueError(f"unknown composition {composition!r}")

    return _mask


def concat_filtered_pooled(
        caches: list[dict], layer: int, pooling: str, mask_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate pooled features across caches, filtered by mask_fn."""
    feats, labs, dids = [], [], []
    for c in caches:
        mask = mask_fn(c)
        if not mask.any():
            continue
        feats.append(_pooled_rows(c, layer, pooling, mask))
        labs.append(c["labels"][mask[c["valid"]]])
        dids.append(np.full(mask.sum(), c["dataset"], dtype=object))
    if not feats:
        return np.empty((0, 0)), np.empty(0), np.empty(0)
    return np.concatenate(feats), np.concatenate(labs), np.concatenate(dids)


def concat_filtered_token(
        caches: list[dict], layer: int, device: str, mask_fn,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate token features across caches, filtered by mask_fn.

    Returns (flat_features, offsets, labels, dataset_ids). Offsets span
    all selected rows across all caches.
    """
    flat_parts, offset_parts, labs, dids = [], [], [], []
    running_total = 0
    for c in caches:
        mask = mask_fn(c)
        if not mask.any():
            continue
        feat, off = _token_rows(c, layer, device, mask)
        shifted = off + running_total
        offset_parts.append(shifted if not offset_parts else shifted[1:])
        flat_parts.append(feat)
        labs.append(c["labels"][mask[c["valid"]]])
        dids.append(np.full(int(mask.sum()), c["dataset"], dtype=object))
        running_total += feat.shape[0]
    if not flat_parts:
        return (torch.empty(0, device=device), np.empty(0, dtype=np.int64),
                np.empty(0), np.empty(0))
    return (torch.cat(flat_parts, dim=0), np.concatenate(offset_parts),
            np.concatenate(labs), np.concatenate(dids))


def run_pooled_probe(
        caches: list[dict], composition: str, probe_name: str, seed: int,
) -> dict | None:
    """Train logistic or MLP on pooled L46 concat features.
    Caches must be filtered to one base model family already."""
    trn_mask_fn = mask_fn_for_composition(composition)
    tst_mask_fn = lambda c: c["test_mask"]

    X_tr, y_tr, _ = concat_filtered_pooled(caches, LAYER, POOLING, trn_mask_fn)
    if X_tr.shape[0] == 0 or not has_both_classes(y_tr):
        print("no training data or single class")
        return None

    if probe_name == "logistic":
        pipe = fit_logistic(X_tr, y_tr, regularization=1.0)
    elif probe_name == "mlp-512":
        pipe = fit_mlp_local(X_tr, y_tr, (512,), alpha=1e-3, seed=seed)
    else:
        raise ValueError(f"unknown pooled probe {probe_name!r}")

    X_te, y_te, ds_te = concat_filtered_pooled(caches, LAYER, POOLING, tst_mask_fn)
    if X_te.shape[0] == 0:
        return None

    scores = pipe.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, scores)
    return dict(
        probe=probe_label(probe_name, (512,)),
        train_data=composition,
        auroc=metrics["auroc"],
        balanced_accuracy=metrics["balanced_accuracy"],
        n_train=len(y_tr),
        n_eval=len(y_te),
        _per_dataset=per_dataset_metrics(
            y_te, scores, ds_te,
            probe=probe_label(probe_name, (512,)), train_data=composition),
    )


def run_token_probe(
        caches: list[dict], composition: str, architecture: str,
        device: str, seed: int,
) -> dict | None:
    """Train a token-sequence probe on L46 token features.
    Caches must be filtered to one base model family already."""
    trn_mask_fn = mask_fn_for_composition(composition)
    tst_mask_fn = lambda c: c["test_mask"]

    X_tr, off_tr, y_tr, _ = concat_filtered_token(
        caches, LAYER, device, trn_mask_fn)
    if X_tr.shape[0] == 0 or not has_both_classes(y_tr):
        print("no training data or single class")
        return None

    probe = TokenProbe(architecture, seed=seed, device=device).fit(
        X_tr, off_tr, y_tr)
    del X_tr

    X_te, off_te, y_te, ds_te = concat_filtered_token(
        caches, LAYER, device, tst_mask_fn)
    if X_te.shape[0] == 0:
        return None

    scores = probe.predict_proba(X_te, off_te)[:, 1]
    metrics = compute_metrics(y_te, scores)
    del X_te

    return dict(
        probe=architecture,
        train_data=composition,
        auroc=metrics["auroc"],
        balanced_accuracy=metrics["balanced_accuracy"],
        n_train=len(y_tr),
        n_eval=len(y_te),
        _per_dataset=per_dataset_metrics(
            y_te, scores, ds_te,
            probe=architecture, train_data=composition),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    parser.add_argument("--pooled-probes", default="logistic,mlp-512")
    parser.add_argument("--token-archs", default="attention,cnn,transformer")
    parser.add_argument("--compositions", default="both,varied,instructed,both+val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load manifest.
    manifest = load_manifest(Path(args.manifest))
    n_tot = len(manifest)
    n_tr = (manifest["dev_split"] == "train").sum()
    n_vl = (manifest["dev_split"] == "validation").sum()
    n_te = (manifest["dev_split"] == "test").sum()
    print(f"Manifest: {n_tot} rows ({n_tr} train / {n_vl} val / {n_te} test)")
    print(f"  Scenarios: {sorted(manifest['scenario'].unique())}")

    # Discover caches.
    caches = discover_standardized_caches(Path(args.cache_dir), manifest)
    if not caches:
        print("No caches with manifest overlap found; nothing to do")
        return

    n_p = sum(1 for c in caches if not c["is_token"])
    n_t = sum(1 for c in caches if c["is_token"])
    print(f"Found {len(caches)} caches ({n_p} pooled, {n_t} token)")
    for c in caches:
        kind = "token" if c["is_token"] else "pooled"
        print(f"  {c['dataset']} ({kind}): {int(c['train_mask'].sum())} train, "
              f"{int(c['test_mask'].sum())} test")

    pooled_probes = args.pooled_probes.split(",")
    token_archs = args.token_archs.split(",")
    compositions = args.compositions.split(",")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    start_time = time.time()

    for composition in compositions:
        print(f"\n=== Composition: {composition} ===")

        for base_model in sorted({c["base_model"] for c in caches}):
            model_caches = [c for c in caches if c["base_model"] == base_model]
            pool_caches = [c for c in model_caches if not c["is_token"]]
            tok_caches = [c for c in model_caches if c["is_token"]]

            for probe_name in pooled_probes:
                if not pool_caches:
                    print(f"  {probe_name} ({base_model}) SKIP (no pooled cache)")
                    continue
                print(f"  {probe_name} ({base_model}) ... ", end="", flush=True)
                result = run_pooled_probe(pool_caches, composition, probe_name, args.seed)
                if result is None:
                    print("SKIP")
                    continue
                result["base_model"] = base_model
                sweep_rows.append({k: v for k, v in result.items() if k != "_per_dataset"})
                for drow in result["_per_dataset"]:
                    drow["base_model"] = base_model
                per_dataset_rows.extend(result["_per_dataset"])
                print(f"BA {result['balanced_accuracy']:.4f}  "
                      f"AUROC {result['auroc']:.4f}  "
                      f"(n={result['n_train']}/{result['n_eval']})")

            for arch in token_archs:
                if not tok_caches:
                    print(f"  {arch} token ({base_model}) SKIP (no token cache)")
                    continue
                print(f"  {arch} token ({base_model}) ... ", end="", flush=True)
                result = run_token_probe(tok_caches, composition, arch, device, args.seed)
                if result is None:
                    print("SKIP")
                    continue
                result["base_model"] = base_model
                sweep_rows.append({k: v for k, v in result.items() if k != "_per_dataset"})
                for drow in result["_per_dataset"]:
                    drow["base_model"] = base_model
                per_dataset_rows.extend(result["_per_dataset"])
                print(f"BA {result['balanced_accuracy']:.4f}  "
                      f"AUROC {result['auroc']:.4f}  "
                      f"(n={result['n_train']}/{result['n_eval']})")

    elapsed = time.time() - start_time
    print(f"\nSweep completed in {elapsed:.0f}s")

    frame = rows_to_frame(sweep_rows, RESULT_COLUMNS)
    frame.to_csv(out_dir / "results.csv", index=False)
    rows_to_frame(per_dataset_rows, PER_DATASET_COLUMNS).to_csv(
        out_dir / "per_dataset.csv", index=False)
    print(f"Wrote {len(frame)} rows to {out_dir / 'results.csv'}")

    # Print summary table.
    print("\n=== Standardized-split results ===")
    header = (f"{'base_model':12s} {'probe':20s} {'train_data':12s} "
              f"{'BA':>8s} {'AUROC':>8s} {'n_train':>7s} {'n_eval':>7s}")
    print(header)
    print("-" * len(header))
    for _, row in frame.iterrows():
        print(f"{row['base_model']:12s} {row['probe']:20s} {row['train_data']:12s} "
              f"{row['balanced_accuracy']:8.4f} {row['auroc']:8.4f} "
              f"{int(row['n_train']):7d} {int(row['n_eval']):7d}")


if __name__ == "__main__":
    main()
