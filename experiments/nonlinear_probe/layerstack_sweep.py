"""
Phase 2 continued: CNN and transformer probes over the cached layer stack.

Instead of one layer's pooled vector, each example is the (layers, hidden)
stack of pooled response activations across every cached decoder layer,
letting the probe learn cross-layer structure. Evaluation protocols match
nonlinear_sweep.py: cross-scenario (both directions, base models with both
scenarios only) and leave-one-organism-out.

Usage:

    python experiments/nonlinear_probe/layerstack_sweep.py
    python experiments/nonlinear_probe/layerstack_sweep.py --mode holdout --archs cnn

Output (under `--out-dir`): the same cross/holdout CSV pairs as
nonlinear_sweep.py, with `layer` set to 'stack' and `probe` to the
architecture name.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from linear_sweep import (
    CacheFile,
    compute_metrics,
    discover_cache_files,
    has_both_classes,
    load_cache_files,
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
from torch_probes import TorchProbe

STACK_LABEL = "stack"


def stacked_features(cache: CacheFile, pooling: str, layer_step: int) -> np.ndarray:
    """
    Read one cache's full layer stack from disk.

    :param cache: Cache metadata (arrays live on disk, see CacheFile).
    :param pooling: 'mean' or 'last'.
    :param layer_step: Keep every n-th cached layer (1 = all).
    :return: (N, layers, hidden) float16 stack restricted to rows with
        usable labels; float16 overflows (gemma's late layers) are clipped
        back to the finite range.
    """
    # Load every cached layer's pooled features and stack them along a new
    # layer axis: (N, layers, hidden). Apply layer_step to thin out the stack.
    layers = sorted(cache.layers)[::layer_step]
    float16_max = np.float16(np.finfo(np.float16).max)
    with np.load(cache.path, allow_pickle=True) as data:
        parts = [
            np.nan_to_num(np.asarray(data[f"{pooling}_L{layer}"])[cache.label_mask],
                          nan=np.float16(0.0), posinf=float16_max, neginf=-float16_max)
            for layer in layers
        ]
    return np.stack(parts, axis=1)


def concat_stacked_features(
        cache_files: list[CacheFile], pooling: str, layer_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param cache_files: Caches to concatenate, in the given order.
    :param pooling: 'mean' or 'last'.
    :param layer_step: Keep every n-th cached layer.
    :return: (features, labels, dataset_ids) with rows in cache_files order.
    """
    features = np.concatenate(
        [stacked_features(cache, pooling, layer_step) for cache in cache_files])
    labels = np.concatenate([cache.labels for cache in cache_files])
    dataset_ids = np.concatenate([
        np.full(len(cache.labels), cache.dataset, dtype=object) for cache in cache_files
    ])
    return features, labels, dataset_ids


def run_cross_scenario(
        cache_files: list[CacheFile], architectures: list[str], pooling: str,
        layer_step: int, seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Train on one scenario's layer stacks and evaluate on the other, per
    architecture; base models with a single scenario are skipped.

    :param cache_files: All loaded caches.
    :param architectures: Torch probe architectures to train.
    :param pooling: Pooling of the per-layer vectors.
    :param layer_step: Keep every n-th cached layer.
    :param seed: Torch seed.
    :return: (cross_rows, per_dataset_rows) as lists of plain dicts.
    """
    cross_rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    for base_model in sorted({cache.base_model for cache in cache_files}):
        files_for_model = [cache for cache in cache_files if cache.base_model == base_model]
        scenarios = scenarios_of(files_for_model)
        # Base models with only one scenario cannot do cross-scenario eval.
        if len(scenarios) < 2:
            print(f"note: {base_model} has only scenario(s) {scenarios}; skipping cross-scenario")
            continue
        for train_scenario in scenarios:
            train_files = [cache for cache in files_for_model
                           if cache.scenario == train_scenario]
            eval_scenario = next(scenario for scenario in scenarios
                                 if scenario != train_scenario)
            eval_files = [cache for cache in files_for_model
                          if cache.scenario == eval_scenario]
            train_features, train_labels, _ = concat_stacked_features(
                train_files, pooling, layer_step)
            eval_features, eval_labels, eval_dataset_ids = concat_stacked_features(
                eval_files, pooling, layer_step)
            if not has_both_classes(train_labels):
                print(f"warning: skipping {base_model}/{train_scenario} (single-class train set)")
                continue
            # Fit every architecture on the same stacked features.
            for architecture in architectures:
                probe = TorchProbe(architecture, seed=seed).fit(train_features, train_labels)
                scores = probe.predict_proba(eval_features)[:, 1]
                metrics = compute_metrics(eval_labels, scores)
                config_fields = dict(base_model=base_model, probe=architecture,
                                     layer=STACK_LABEL, pooling=pooling,
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
                print(f"cross {base_model}/{architecture} {train_scenario}->{eval_scenario}: "
                      f"BA {metrics['balanced_accuracy']:.3f} AUROC {metrics['auroc']:.3f}",
                      flush=True)
    return cross_rows, per_dataset_rows


def run_holdout(
        cache_files: list[CacheFile], architectures: list[str], pooling: str,
        layer_step: int, seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Leave-one-organism-out over layer stacks, per architecture.

    :param cache_files: All loaded caches.
    :param architectures: Torch probe architectures to train.
    :param pooling: Pooling of the per-layer vectors.
    :param layer_step: Keep every n-th cached layer.
    :param seed: Torch seed.
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
        # Train on all organisms except one, evaluate on the held-out one.
        for held_out, architecture in product(organisms, architectures):
            train_files = [cache for cache in files_for_model
                           if parse_organism(cache.dataset, cache.model_id) != held_out]
            eval_files = [cache for cache in files_for_model
                          if parse_organism(cache.dataset, cache.model_id) == held_out]
            train_features, train_labels, _ = concat_stacked_features(
                train_files, pooling, layer_step)
            eval_features, eval_labels, eval_dataset_ids = concat_stacked_features(
                eval_files, pooling, layer_step)
            if not has_both_classes(train_labels):
                print(f"warning: skipping holdout {base_model}/{held_out} "
                      f"(single-class train set)")
                continue
            probe = TorchProbe(architecture, seed=seed).fit(train_features, train_labels)
            scores = probe.predict_proba(eval_features)[:, 1]
            metrics = compute_metrics(eval_labels, scores)
            config_fields = dict(base_model=base_model, probe=architecture,
                                 layer=STACK_LABEL, pooling=pooling,
                                 holdout_organism=held_out)
            holdout_rows.append({
                **config_fields,
                "auroc": metrics["auroc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n_train": len(train_labels), "n_eval": len(eval_labels),
            })
            per_dataset_rows.extend(per_dataset_metrics(
                eval_labels, scores, eval_dataset_ids, **config_fields))
            print(f"holdout {base_model}/{architecture} held-out={held_out}: "
                  f"BA {metrics['balanced_accuracy']:.3f} AUROC {metrics['auroc']:.3f}",
                  flush=True)
    return holdout_rows, per_dataset_rows


def main() -> None:
    """
    Parse CLI arguments, run the requested evaluation modes, and write
    results to `--out-dir`.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache-dir", default="results/whitebox/activations")
    parser.add_argument("--out-dir", default="results/whitebox/layerstack_sweep")
    parser.add_argument("--mode", choices=("cross", "holdout", "both"), default="both")
    parser.add_argument("--archs", default="cnn,transformer",
                        help="comma-separated torch probe architectures")
    parser.add_argument("--pooling", default="mean", choices=("mean", "last"))
    parser.add_argument("--layer-step", type=int, default=1,
                        help="keep every n-th cached layer in the stack")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-limit", action="store_true",
                        help="also load '.limit' smoke caches (for pipeline smoke tests)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    paths = discover_cache_files(cache_dir, include_limit=args.include_limit)
    print(f"found {len(paths)} cache file(s) in {cache_dir} (include_limit={args.include_limit})")
    cache_files = load_cache_files(paths)
    if not cache_files:
        print("no usable cache files found; nothing to do")
        return

    architectures = args.archs.split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cross-scenario: train stacked-feature probes on one scenario, evaluate
    # on the other.
    if args.mode in ("cross", "both"):
        cross_rows, cross_per_dataset = run_cross_scenario(
            cache_files, architectures, args.pooling, args.layer_step, args.seed)
        cross_frame = rows_to_frame(cross_rows, CROSS_COLUMNS)
        cross_frame.to_csv(out_dir / "cross_results.csv", index=False)
        rows_to_frame(cross_per_dataset, CROSS_PER_DATASET_COLUMNS).to_csv(
            out_dir / "cross_per_dataset.csv", index=False)
        print(f"wrote {len(cross_frame)} cross-scenario rows to {out_dir}")

    # Leave-one-organism-out: train on all organisms except one, evaluate
    # on the held-out organism's datasets.
    if args.mode in ("holdout", "both"):
        holdout_rows, holdout_per_dataset = run_holdout(
            cache_files, architectures, args.pooling, args.layer_step, args.seed)
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
