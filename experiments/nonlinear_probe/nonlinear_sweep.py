"""
Phase 2: nonlinear-probe sweep over cached residual-stream activations.

Compares an MLP probe against the Phase 1 logistic baseline on the same
activation caches, under two evaluation protocols:

- cross-scenario (as in linear_sweep.py): train on all datasets of one
  scenario, evaluate on all datasets of the other, both directions. Only
  runs for base models with both scenarios cached (Qwen); gemma's CV
  fallback lives in linear_sweep.py and is not repeated here.
- leave-one-organism-out (holdout): train on all datasets of both scenarios
  except the held-out organism's, evaluate on the held-out organism's
  datasets. This tests generalization to unseen model organisms, which
  cross-scenario evaluation cannot (the same organisms appear in both
  scenarios).

Usage:

    python experiments/nonlinear_probe/nonlinear_sweep.py
    python experiments/nonlinear_probe/nonlinear_sweep.py --mode holdout --holdout-layers 42,46

Output (under `--out-dir`):
    cross_results.csv     one row per (base_model, probe, layer, pooling,
                          train_scenario, eval_scenario)
    cross_per_dataset.csv the same, one row per eval dataset
    holdout_results.csv   one row per (base_model, probe, layer, pooling,
                          holdout_organism)
    holdout_per_dataset.csv the same, one row per held-out dataset
"""

from __future__ import annotations

import argparse
import re
from itertools import product
from pathlib import Path

import pandas as pd

from linear_sweep import (
    CacheFile,
    compute_metrics,
    concat_cache_features,
    discover_cache_files,
    fit_logistic,
    has_both_classes,
    load_cache_files,
    parse_layers_arg,
    per_dataset_metrics,
    rows_to_frame,
)

CROSS_COLUMNS = [
    "base_model", "probe", "layer", "pooling", "train_scenario", "eval_scenario",
    "auroc", "balanced_accuracy", "n_train", "n_eval",
]
CROSS_PER_DATASET_COLUMNS = [
    "base_model", "probe", "layer", "pooling", "train_scenario", "eval_scenario",
    "dataset", "auroc", "balanced_accuracy", "n_eval",
]
HOLDOUT_COLUMNS = [
    "base_model", "probe", "layer", "pooling", "holdout_organism",
    "auroc", "balanced_accuracy", "n_train", "n_eval",
]
HOLDOUT_PER_DATASET_COLUMNS = [
    "base_model", "probe", "layer", "pooling", "holdout_organism",
    "dataset", "auroc", "balanced_accuracy", "n_eval",
]


def parse_organism(dataset_name: str, model_id: str) -> str:
    """
    Extract a normalized model-organism label from a dev dataset name.

    :param dataset_name: Dataset id such as
        'aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3'.
    :param model_id: HF base-model id from the cache meta, e.g.
        'Qwen/Qwen3.5-27B'; its name part separates base model from organism.
    :return: Organism label with the base-model echo stripped, e.g. 'a-mo-3',
        'b-mo', 'g-st-2', or 'None' for the plain base model.
    :raises ValueError: If the dataset name does not contain the model token.
    """
    # Strip the base-model echo that some datasets append to the organism tag.
    model_token = model_id.split("/")[-1]
    match = re.search(rf"deception-{re.escape(model_token)}-(?P<organism>.+)$", dataset_name)
    if match is None:
        raise ValueError(
            f"could not parse organism from {dataset_name!r} with model token {model_token!r}")
    raw_organism = match.group("organism")
    return raw_organism.replace(f"-{model_token.lower()}", "")


def probe_label(probe: str, hidden_layers: tuple[int, ...]) -> str:
    """
    :param probe: 'logistic' or 'mlp'.
    :param hidden_layers: MLP hidden layer sizes (ignored for logistic).
    :return: Label used in result rows, e.g. 'logistic' or 'mlp-512'.
    """
    if probe == "logistic":
        return "logistic"
    return "mlp-" + "x".join(str(size) for size in hidden_layers)


def parse_hidden_arg(spec: str) -> tuple[int, ...]:
    """
    :param spec: Comma-separated hidden layer sizes, e.g. '512' or '512,128'.
    :return: Tuple of hidden layer sizes for MLPClassifier.
    """
    return tuple(int(part) for part in spec.split(","))


def fit_mlp(features, labels, hidden_layers: tuple[int, ...], alpha: float, seed: int):
    """
    :param features: (N, D) float32 training features.
    :param labels: (N,) binary training labels.
    :param hidden_layers: Hidden layer sizes.
    :param alpha: L2 penalty strength.
    :param seed: Weight-init and validation-split seed.
    :return: A fitted scaler + MLPClassifier sklearn Pipeline (Adam, early
        stopping on a held-out 15% validation split).
    """
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


def fit_probe(features, labels, probe: str, hidden_layers: tuple[int, ...],
              alpha: float, regularization: float, seed: int):
    """
    :param features: (N, D) float32 training features.
    :param labels: (N,) binary training labels.
    :param probe: 'logistic' (Phase 1 baseline) or 'mlp'.
    :param hidden_layers: MLP hidden layer sizes (ignored for logistic).
    :param alpha: MLP L2 penalty (ignored for logistic).
    :param regularization: Logistic inverse regularization C (ignored for mlp).
    :param seed: MLP seed (ignored for logistic).
    :return: A fitted sklearn Pipeline.
    :raises ValueError: On an unknown probe name.
    """
    if probe == "logistic":
        return fit_logistic(features, labels, regularization)
    if probe == "mlp":
        return fit_mlp(features, labels, hidden_layers, alpha, seed)
    raise ValueError(f"unknown probe {probe!r}")


def scenarios_of(cache_files: list[CacheFile]) -> list[str]:
    """
    :param cache_files: Loaded caches for one base model.
    :return: Sorted distinct scenario labels present.
    """
    return sorted({cache.scenario for cache in cache_files})


def common_layers(cache_files: list[CacheFile], layers_override: list[int] | None) -> list[int]:
    """
    :param cache_files: Loaded caches for one base model.
    :param layers_override: If given, intersect with the cached layers.
    :return: Sorted layer indices available in every cache.
    """
    shared = set.intersection(*(set(cache.layers) for cache in cache_files))
    if layers_override is not None:
        shared &= set(layers_override)
    return sorted(shared)


def run_cross_scenario(
        cache_files: list[CacheFile], probes: list[str], layers_override: list[int] | None,
        poolings: list[str], hidden_layers: tuple[int, ...], alpha: float,
        regularization: float, seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Train on one scenario and evaluate on the other, per probe type.

    :param cache_files: All loaded caches.
    :param probes: Probe names to fit on each shared feature load.
    :param layers_override: Restrict to these layer indices if given.
    :param poolings: Pooling strategies to sweep.
    :param hidden_layers: MLP hidden layer sizes.
    :param alpha: MLP L2 penalty.
    :param regularization: Logistic inverse regularization C.
    :param seed: MLP seed.
    :return: (cross_rows, per_dataset_rows) as lists of plain dicts. Base
        models with only one cached scenario are skipped with a note.
    """
    cross_rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    base_models = sorted({cache.base_model for cache in cache_files})
    for base_model in base_models:
        files_for_model = [cache for cache in cache_files if cache.base_model == base_model]
        scenarios = scenarios_of(files_for_model)
        # Skip base models that only have one scenario (their CV numbers are
        # already in linear_sweep.py and not repeated here).
        if len(scenarios) < 2:
            print(f"note: {base_model} has only scenario(s) {scenarios}; "
                  f"skipping cross-scenario (see linear_sweep.py for its CV numbers)")
            continue
        layers = common_layers(files_for_model, layers_override)
        for layer, pooling, train_scenario in product(layers, poolings, scenarios):
            train_files = [cache for cache in files_for_model if cache.scenario == train_scenario]
            eval_scenarios = [scenario for scenario in scenarios if scenario != train_scenario]
            train_features, train_labels, _ = concat_cache_features(train_files, layer, pooling)
            if not has_both_classes(train_labels):
                print(f"warning: skipping {base_model}/L{layer}/{pooling}/{train_scenario} "
                      f"(single-class train set)")
                continue
            # Fit every probe on the same train data, evaluate on the held-out scenario.
            for eval_scenario in eval_scenarios:
                eval_files = [cache for cache in files_for_model
                              if cache.scenario == eval_scenario]
                eval_features, eval_labels, eval_dataset_ids = concat_cache_features(
                    eval_files, layer, pooling)
                for probe in probes:
                    pipeline = fit_probe(train_features, train_labels, probe,
                                         hidden_layers, alpha, regularization, seed)
                    scores = pipeline.predict_proba(eval_features)[:, 1]
                    metrics = compute_metrics(eval_labels, scores)
                    config_fields = dict(
                        base_model=base_model, probe=probe_label(probe, hidden_layers),
                        layer=layer, pooling=pooling, train_scenario=train_scenario)
                    cross_rows.append({
                        **config_fields, "eval_scenario": eval_scenario,
                        "auroc": metrics["auroc"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "n_train": len(train_labels), "n_eval": len(eval_labels),
                    })
                    per_dataset_rows.extend(per_dataset_metrics(
                        eval_labels, scores, eval_dataset_ids,
                        **config_fields, eval_scenario=eval_scenario))
                    print(f"cross {base_model}/{config_fields['probe']}/L{layer}/{pooling} "
                          f"{train_scenario}->{eval_scenario}: "
                          f"BA {metrics['balanced_accuracy']:.3f} AUROC {metrics['auroc']:.3f}",
                          flush=True)
    return cross_rows, per_dataset_rows


def run_holdout(
        cache_files: list[CacheFile], probes: list[str], layers_override: list[int] | None,
        poolings: list[str], hidden_layers: tuple[int, ...], alpha: float,
        regularization: float, seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Leave-one-organism-out: hold out every organism in turn, train on the
    rest (both scenarios pooled), evaluate on the held-out organism.

    :param cache_files: All loaded caches.
    :param probes: Probe names to fit on each shared feature load.
    :param layers_override: Restrict to these layer indices if given.
    :param poolings: Pooling strategies to sweep.
    :param hidden_layers: MLP hidden layer sizes.
    :param alpha: MLP L2 penalty.
    :param regularization: Logistic inverse regularization C.
    :param seed: MLP seed.
    :return: (holdout_rows, per_dataset_rows) as lists of plain dicts. Base
        models with fewer than two organisms are skipped.
    """
    holdout_rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    base_models = sorted({cache.base_model for cache in cache_files})
    for base_model in base_models:
        files_for_model = [cache for cache in cache_files if cache.base_model == base_model]
        organisms = sorted({parse_organism(cache.dataset, cache.model_id)
                            for cache in files_for_model})
        if len(organisms) < 2:
            print(f"note: {base_model} has only organism(s) {organisms}; skipping holdout")
            continue
        layers = common_layers(files_for_model, layers_override)
        # Hold out one organism at a time; train on everything else (both
        # scenarios pooled), evaluate on the held-out organism's datasets.
        for layer, pooling, held_out in product(layers, poolings, organisms):
            train_files = [cache for cache in files_for_model
                           if parse_organism(cache.dataset, cache.model_id) != held_out]
            eval_files = [cache for cache in files_for_model
                          if parse_organism(cache.dataset, cache.model_id) == held_out]
            train_features, train_labels, _ = concat_cache_features(train_files, layer, pooling)
            eval_features, eval_labels, eval_dataset_ids = concat_cache_features(
                eval_files, layer, pooling)
            if not has_both_classes(train_labels):
                print(f"warning: skipping holdout {base_model}/L{layer}/{pooling}/{held_out} "
                      f"(single-class train set)")
                continue
            for probe in probes:
                pipeline = fit_probe(train_features, train_labels, probe,
                                     hidden_layers, alpha, regularization, seed)
                scores = pipeline.predict_proba(eval_features)[:, 1]
                metrics = compute_metrics(eval_labels, scores)
                config_fields = dict(
                    base_model=base_model, probe=probe_label(probe, hidden_layers),
                    layer=layer, pooling=pooling, holdout_organism=held_out)
                holdout_rows.append({
                    **config_fields,
                    "auroc": metrics["auroc"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "n_train": len(train_labels), "n_eval": len(eval_labels),
                })
                per_dataset_rows.extend(per_dataset_metrics(
                    eval_labels, scores, eval_dataset_ids, **config_fields))
                print(f"holdout {base_model}/{config_fields['probe']}/L{layer}/{pooling} "
                      f"held-out={held_out}: BA {metrics['balanced_accuracy']:.3f} "
                      f"AUROC {metrics['auroc']:.3f}", flush=True)
    return holdout_rows, per_dataset_rows


def holdout_summary(holdout_frame: pd.DataFrame) -> pd.DataFrame:
    """
    :param holdout_frame: The holdout_results rows.
    :return: Pivot of balanced accuracy with one row per (base_model, layer,
        pooling, holdout_organism) and one column per probe, plus a mean row
        per probe at the bottom of each config group, for quick comparison.
    """
    if holdout_frame.empty:
        return pd.DataFrame()
    return holdout_frame.pivot_table(
        index=["base_model", "layer", "pooling", "holdout_organism"],
        columns="probe", values="balanced_accuracy",
    ).round(3)


def main() -> None:
    """
    Parse CLI arguments, run the requested evaluation modes, and write
    results to `--out-dir`.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache-dir", default="results/whitebox/activations")
    parser.add_argument("--out-dir", default="results/whitebox/nonlinear_sweep")
    parser.add_argument("--mode", choices=("cross", "holdout", "both"), default="both")
    parser.add_argument("--probes", default="logistic,mlp",
                        help="comma-separated probe types to fit per config")
    parser.add_argument("--cross-layers", default="36,40,44,46,48",
                        help="layer indices for the cross-scenario sweep")
    parser.add_argument("--cross-poolings", default="mean,last,concat")
    parser.add_argument("--holdout-layers", default="46",
                        help="layer indices for the leave-one-organism-out sweep")
    parser.add_argument("--holdout-poolings", default="concat")
    parser.add_argument("--hidden", default="512",
                        help="comma-separated MLP hidden layer sizes, e.g. '512' or '512,128'")
    parser.add_argument("--alpha", type=float, default=1e-3, help="MLP L2 penalty")
    parser.add_argument("--C", dest="regularization", type=float, default=1.0,
                        help="inverse regularization strength for the logistic baseline")
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

    probes = args.probes.split(",")
    hidden_layers = parse_hidden_arg(args.hidden)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cross-scenario: train on one scenario, evaluate on the other (both
    # directions), comparing logistic vs MLP on the same train/eval splits.
    if args.mode in ("cross", "both"):
        cross_rows, cross_per_dataset = run_cross_scenario(
            cache_files, probes, parse_layers_arg(args.cross_layers),
            args.cross_poolings.split(","), hidden_layers, args.alpha,
            args.regularization, args.seed)
        cross_frame = rows_to_frame(cross_rows, CROSS_COLUMNS)
        cross_frame.to_csv(out_dir / "cross_results.csv", index=False)
        rows_to_frame(cross_per_dataset, CROSS_PER_DATASET_COLUMNS).to_csv(
            out_dir / "cross_per_dataset.csv", index=False)
        print(f"wrote {len(cross_frame)} cross-scenario rows to {out_dir}")

    # Leave-one-organism-out: train on all organisms except one, evaluate on
    # the held-out organism's datasets (both scenarios pooled for training).
    if args.mode in ("holdout", "both"):
        holdout_rows, holdout_per_dataset = run_holdout(
            cache_files, probes, parse_layers_arg(args.holdout_layers),
            args.holdout_poolings.split(","), hidden_layers, args.alpha,
            args.regularization, args.seed)
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
