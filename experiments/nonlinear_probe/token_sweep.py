"""
Per-token probe sweep over cached response-token activations.

Instead of a single pooled vector per example (linear_sweep.py,
nonlinear_sweep.py) or a stack of pooled vectors across layers
(layerstack_sweep.py), each example here is its full (tokens, hidden)
sequence of response-token activations at one layer, letting the probe
learn which tokens carry the deception signal instead of relying on a
fixed mean/last pooling. Evaluation protocols match layerstack_sweep.py:
cross-scenario (both directions, base models with both scenarios only) and
leave-one-organism-out.

Usage:

    python experiments/nonlinear_probe/token_sweep.py
    python experiments/nonlinear_probe/token_sweep.py --mode holdout --archs attention

Output (under `--out-dir`): the same cross/holdout CSV pairs as
nonlinear_sweep.py, with `layer` set to the cached layer index and `pooling`
to 'tokens'.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from linear_sweep import (
    base_model_family,
    compute_metrics,
    has_both_classes,
    is_limit_cache,
    meta_model_id,
    parse_layers_arg,
    parse_scenario,
    per_dataset_metrics,
    rows_to_frame,
)
from nonlinear_sweep import (
    CROSS_COLUMNS,
    CROSS_PER_DATASET_COLUMNS,
    HOLDOUT_COLUMNS,
    HOLDOUT_PER_DATASET_COLUMNS,
    holdout_summary,
    parse_organism,
    scenarios_of,
)
from token_probes import TokenProbe

TOKENS_POOLING_LABEL = "tokens"
CACHE_GLOB = "*.tokens.npz"


@dataclass
class TokenCacheFile:
    """
    Metadata for one per-token activation cache. Token arrays are
    deliberately not held here: reading every layer of every cache up front
    would exceed RAM, so `load_token_features` re-opens the npz on demand for
    a single requested layer.
    """

    path: Path
    dataset: str
    scenario: str
    base_model: str
    model_id: str
    layers: list[int]
    labels: np.ndarray


def discover_token_cache_files(cache_dir: Path, include_limit: bool = False) -> list[Path]:
    """
    :param cache_dir: Directory containing `*.tokens.npz` activation caches.
    :param include_limit: If True, also include `.limit` smoke caches;
        otherwise they are skipped (the default for a real sweep).
    :return: Sorted list of cache file paths to load.
    """
    paths = sorted(cache_dir.glob(CACHE_GLOB))
    if include_limit:
        return paths
    return [path for path in paths if not is_limit_cache(path.name)]


def load_token_cache_file(path: Path) -> TokenCacheFile | None:
    """
    Load one token-cache's metadata and validate it has labels.

    :param path: Path to a `.tokens.npz` cache.
    :return: A populated TokenCacheFile, or None if the cache has no
        `deceptive` labels (a warning is printed to stdout in that case).
    """
    with np.load(path, allow_pickle=True) as data:
        if "deceptive" not in data:
            print(f"warning: {path.name} has no labels; skipping")
            return None
        meta = json.loads(str(data["meta"]))
        dataset = meta["dataset"]
        return TokenCacheFile(
            path=path,
            dataset=dataset,
            scenario=parse_scenario(dataset),
            base_model=base_model_family(meta_model_id(meta)),
            model_id=meta_model_id(meta),
            layers=list(meta["layers"]),
            labels=np.asarray(data["deceptive"]).astype(np.int64),
        )


def load_token_cache_files(paths: Iterable[Path]) -> list[TokenCacheFile]:
    """
    :param paths: Cache file paths to load.
    :return: Loaded caches, dropping any with no usable labels.
    """
    loaded = []
    for path in paths:
        cache = load_token_cache_file(path)
        if cache is not None:
            loaded.append(cache)
    return loaded


def load_token_features(
        cache: TokenCacheFile, layer: int, device: str,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Read one cache's single-layer token features from disk.

    :param cache: Cache metadata (arrays live on disk, see TokenCacheFile).
    :param layer: Decoder layer index; reads the `tokens_L{layer}` array.
    :param device: Torch device the returned feature tensor is moved to.
    :return: (flat_features, offsets): flat_features is (total_tokens,
        hidden) float16 on `device`; offsets is (N+1,) int64 with example
        i's tokens at flat_features[offsets[i]:offsets[i + 1]].
    """
    with np.load(cache.path, allow_pickle=True) as data:
        flat_features = torch.from_numpy(np.asarray(data[f"tokens_L{layer}"])).to(device)
        offsets = np.asarray(data["token_offsets"], dtype=np.int64)
    # Clamp infinities (float16 overflow artifacts, see gemma L46) so they
    # don't produce NaN during standardisation or training.
    finfo = torch.finfo(flat_features.dtype)
    flat_features = flat_features.clamp(finfo.min, finfo.max)
    return flat_features, offsets


def concat_token_features(
        cache_files: list[TokenCacheFile], layer: int, device: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge several caches' single-layer token features into one flat tensor.

    :param cache_files: Caches to concatenate, in the given order.
    :param layer: Decoder layer index to read from every cache.
    :param device: Torch device the merged feature tensor lives on.
    :return: (flat_features, offsets, labels, dataset_ids). flat_features is
        (total_tokens, hidden) on `device`; offsets is (sum(N) + 1,) int64
        spanning every cache (each cache's offsets are shifted by the
        running token total before merging); labels and dataset_ids are one
        entry per example, in cache_files order.
    """
    flat_features_parts = []
    offset_parts = []
    label_parts = []
    dataset_id_parts = []
    running_token_total = 0
    for cache in cache_files:
        flat_features, offsets = load_token_features(cache, layer, device)
        # Shift each cache's offsets by the running token total, skipping
        # the first element (which is always 0) for all but the first cache.
        shifted_offsets = offsets + running_token_total
        offset_parts.append(shifted_offsets if not offset_parts else shifted_offsets[1:])
        flat_features_parts.append(flat_features)
        label_parts.append(cache.labels)
        dataset_id_parts.append(np.full(len(cache.labels), cache.dataset, dtype=object))
        running_token_total += flat_features.shape[0]
    merged_features = torch.cat(flat_features_parts, dim=0)
    merged_offsets = np.concatenate(offset_parts)
    merged_labels = np.concatenate(label_parts)
    merged_dataset_ids = np.concatenate(dataset_id_parts)
    return merged_features, merged_offsets, merged_labels, merged_dataset_ids


def run_cross_scenario(
        cache_files: list[TokenCacheFile], architectures: list[str], layers: list[int],
        device: str, seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Train on one scenario's token sequences and evaluate on the other, per
    layer and architecture; base models with a single scenario are skipped.

    :param cache_files: All loaded caches.
    :param architectures: Token probe architectures to train.
    :param layers: Decoder layer indices to sweep.
    :param device: Torch device for feature tensors and training.
    :param seed: Torch/probe seed.
    :return: (cross_rows, per_dataset_rows) as lists of plain dicts.
    """
    cross_rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    for base_model in sorted({cache.base_model for cache in cache_files}):
        files_for_model = [cache for cache in cache_files if cache.base_model == base_model]
        scenarios = scenarios_of(files_for_model)
        if len(scenarios) < 2:
            print(f"note: {base_model} has only scenario(s) {scenarios}; skipping cross-scenario")
            continue
        for layer, train_scenario in product(layers, scenarios):
            train_files = [cache for cache in files_for_model
                           if cache.scenario == train_scenario]
            eval_scenario = next(scenario for scenario in scenarios
                                 if scenario != train_scenario)
            eval_files = [cache for cache in files_for_model
                          if cache.scenario == eval_scenario]
            train_features, train_offsets, train_labels, _ = concat_token_features(
                train_files, layer, device)
            eval_features, eval_offsets, eval_labels, eval_dataset_ids = concat_token_features(
                eval_files, layer, device)
            if not has_both_classes(train_labels):
                print(f"warning: skipping {base_model}/L{layer}/{train_scenario} "
                      f"(single-class train set)")
                continue
            # Fit every token-probe architecture on the same flat features.
            for architecture in architectures:
                probe = TokenProbe(architecture, seed=seed, device=device).fit(
                    train_features, train_offsets, train_labels)
                scores = probe.predict_proba(eval_features, eval_offsets)[:, 1]
                metrics = compute_metrics(eval_labels, scores)
                config_fields = dict(base_model=base_model, probe=architecture,
                                     layer=layer, pooling=TOKENS_POOLING_LABEL,
                                     train_scenario=train_scenario)
                cross_rows.append({
                    **config_fields, "eval_scenario": eval_scenario,
                    "auroc": metrics["auroc"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "n_train": len(train_labels), "n_eval": len(eval_labels),
                })
                per_dataset_rows.extend(per_dataset_metrics(
                    eval_labels, scores, eval_dataset_ids,
                    **config_fields, eval_scenario=eval_scenario))
                print(f"cross {base_model}/{architecture}/L{layer} "
                      f"{train_scenario}->{eval_scenario}: "
                      f"BA {metrics['balanced_accuracy']:.3f} AUROC {metrics['auroc']:.3f}",
                      flush=True)
            # Free GPU memory before the next (layer, scenario) iteration.
            del train_features, eval_features
    return cross_rows, per_dataset_rows


def run_holdout(
        cache_files: list[TokenCacheFile], architectures: list[str], layers: list[int],
        device: str, seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Leave-one-organism-out over token sequences, per layer and architecture.

    :param cache_files: All loaded caches.
    :param architectures: Token probe architectures to train.
    :param layers: Decoder layer indices to sweep.
    :param device: Torch device for feature tensors and training.
    :param seed: Torch/probe seed.
    :return: (holdout_rows, per_dataset_rows) as lists of plain dicts.
    """
    holdout_rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    for base_model in sorted({cache.base_model for cache in cache_files}):
        files_for_model = [cache for cache in cache_files if cache.base_model == base_model]
        organisms = sorted({parse_organism(cache.dataset, cache.model_id)
                            for cache in files_for_model})
        if len(organisms) < 2:
            print(f"note: {base_model} has only organism(s) {organisms}; skipping holdout")
            continue
        for layer, held_out, architecture in product(layers, organisms, architectures):
            train_files = [cache for cache in files_for_model
                           if parse_organism(cache.dataset, cache.model_id) != held_out]
            eval_files = [cache for cache in files_for_model
                          if parse_organism(cache.dataset, cache.model_id) == held_out]
            train_features, train_offsets, train_labels, _ = concat_token_features(
                train_files, layer, device)
            eval_features, eval_offsets, eval_labels, eval_dataset_ids = concat_token_features(
                eval_files, layer, device)
            if not has_both_classes(train_labels):
                print(f"warning: skipping holdout {base_model}/L{layer}/{held_out} "
                      f"(single-class train set)")
                del train_features, eval_features
                continue
            probe = TokenProbe(architecture, seed=seed, device=device).fit(
                train_features, train_offsets, train_labels)
            scores = probe.predict_proba(eval_features, eval_offsets)[:, 1]
            metrics = compute_metrics(eval_labels, scores)
            config_fields = dict(base_model=base_model, probe=architecture,
                                 layer=layer, pooling=TOKENS_POOLING_LABEL,
                                 holdout_organism=held_out)
            holdout_rows.append({
                **config_fields,
                "auroc": metrics["auroc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n_train": len(train_labels), "n_eval": len(eval_labels),
            })
            per_dataset_rows.extend(per_dataset_metrics(
                eval_labels, scores, eval_dataset_ids, **config_fields))
            print(f"holdout {base_model}/{architecture}/L{layer} held-out={held_out}: "
                  f"BA {metrics['balanced_accuracy']:.3f} AUROC {metrics['auroc']:.3f}",
                  flush=True)
            # Free GPU memory before the next (layer, organism, architecture) iteration.
            del train_features, eval_features
    return holdout_rows, per_dataset_rows


def main() -> None:
    """
    Parse CLI arguments, run the requested evaluation modes, and write
    results to `--out-dir`.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache-dir", default="results/whitebox/activations")
    parser.add_argument("--out-dir", default="results/whitebox/token_sweep")
    parser.add_argument("--mode", choices=("cross", "holdout", "both"), default="both")
    parser.add_argument("--archs", default="attention,cnn,transformer",
                        help="comma-separated token probe architectures")
    parser.add_argument("--layers", default="46",
                        help="comma-separated decoder layer indices to sweep")
    parser.add_argument("--device", default=None,
                        help="torch device for feature tensors and training "
                             "(default: cuda if available, else cpu)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-limit", action="store_true",
                        help="also load '.limit' smoke caches (for pipeline smoke tests)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    paths = discover_token_cache_files(cache_dir, include_limit=args.include_limit)
    print(f"found {len(paths)} token cache file(s) in {cache_dir} "
          f"(include_limit={args.include_limit})")
    cache_files = load_token_cache_files(paths)
    if not cache_files:
        print("no usable cache files found; nothing to do")
        return

    architectures = args.archs.split(",")
    layers = parse_layers_arg(args.layers)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cross-scenario: train token-sequence probes on one scenario, evaluate
    # on the other.
    if args.mode in ("cross", "both"):
        cross_rows, cross_per_dataset = run_cross_scenario(
            cache_files, architectures, layers, device, args.seed)
        cross_frame = rows_to_frame(cross_rows, CROSS_COLUMNS)
        cross_frame.to_csv(out_dir / "cross_results.csv", index=False)
        rows_to_frame(cross_per_dataset, CROSS_PER_DATASET_COLUMNS).to_csv(
            out_dir / "cross_per_dataset.csv", index=False)
        print(f"wrote {len(cross_frame)} cross-scenario rows to {out_dir}")

    # Leave-one-organism-out: train on all organisms except one, evaluate
    # on the held-out organism's token sequences.
    if args.mode in ("holdout", "both"):
        holdout_rows, holdout_per_dataset = run_holdout(
            cache_files, architectures, layers, device, args.seed)
        holdout_frame = rows_to_frame(holdout_rows, HOLDOUT_COLUMNS)
        holdout_frame.to_csv(out_dir / "holdout_results.csv", index=False)
        rows_to_frame(holdout_per_dataset, HOLDOUT_PER_DATASET_COLUMNS).to_csv(
            out_dir / "holdout_per_dataset.csv", index=False)
        print(f"wrote {len(holdout_frame)} holdout rows to {out_dir}")
        summary = holdout_summary(holdout_frame)
        if not summary.empty:
            print("balanced accuracy per held-out organism (columns: probe):")
            print(summary.to_string())


if __name__ == "__main__":
    main()
