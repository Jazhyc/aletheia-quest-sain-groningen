"""
Cross-scenario linear-probe sweep over cached residual-stream activations.

Loads every full (non-`.limit`) activation cache written by
`extract_activations.py`, groups caches by base model family (hidden dims
differ across families, so probes can't be shared) and by scenario (`varied`
vs `instructed`), then for every (base model, layer, pooling) trains a
scikit-learn logistic-regression probe on one scenario and evaluates it on
the other, in both directions. When a base model only has caches from one
scenario (e.g. during early data collection), it falls back to a 5-fold
within-scenario cross-validation report instead of skipping the config.

Usage:

    python experiments/nonlinear_probe/linear_sweep.py

    python experiments/nonlinear_probe/linear_sweep.py --include-limit  # smoke test

Output (under `--out-dir`):
    sweep_results.csv   one row per (base_model, layer, pooling, train_scenario,
                         eval_scenario) with auroc, balanced_accuracy, n_train, n_eval
    per_dataset.csv     the same breakdown, one row per eval dataset within
                         each config
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

POOLINGS = ("mean", "last", "concat")
SCENARIOS = ("varied", "instructed")
CV_EVAL_LABEL = "cv"

SWEEP_COLUMNS = [
    "base_model", "layer", "pooling", "train_scenario", "eval_scenario",
    "auroc", "balanced_accuracy", "n_train", "n_eval",
]
PER_DATASET_COLUMNS = [
    "base_model", "layer", "pooling", "train_scenario", "eval_scenario",
    "dataset", "auroc", "balanced_accuracy", "n_eval",
]


def is_limit_cache(filename: str) -> bool:
    """
    :param filename: Cache file name (not full path).
    :return: True if the name marks it a `--limit`-truncated smoke cache.
    """
    return ".limit" in filename


def is_tokens_cache(filename: str) -> bool:
    """
    :param filename: Cache file name (not full path).
    :return: True if the name marks it a `--tokens` per-token cache. These
        hold `tokens_L{L}` arrays instead of `mean_L{L}`/`last_L{L}` pooled
        arrays and would crash the pooled sweep if loaded.
    """
    return ".tokens" in filename


def discover_cache_files(cache_dir: Path, include_limit: bool = False) -> list[Path]:
    """
    :param cache_dir: Directory containing `*.npz` activation caches.
    :param include_limit: If True, also include `.limit` smoke caches;
        otherwise they are skipped (the default for a real sweep). `.tokens`
        per-token caches are always skipped regardless of this flag.
    :return: Sorted list of cache file paths to load.
    """
    paths = sorted(cache_dir.glob("*.npz"))
    paths = [path for path in paths if not is_tokens_cache(path.name)]
    if include_limit:
        return paths
    return [path for path in paths if not is_limit_cache(path.name)]


def parse_scenario(dataset_name: str) -> str:
    """
    :param dataset_name: Dataset id such as
        'aletheias-quest/dev-varied-deception-Qwen3.5-27B-None'.
    :return: The scenario token ('varied' or 'instructed').
    :raises ValueError: If no known scenario token is found.
    """
    match = re.search(r"dev-(varied|instructed)-deception-", dataset_name)
    if match is None:
        raise ValueError(f"could not parse scenario from dataset name {dataset_name!r}")
    return match.group(1)


def base_model_family(model_id: str) -> str:
    """
    :param model_id: HF model id, e.g. 'Qwen/Qwen3.5-27B' or 'google/gemma-3-27b-it'.
    :return: A coarse family label ('qwen', 'gemma', ...) used to group caches
        that share a hidden dimension and can be probed together.
    """
    lowered = model_id.lower()
    for family in ("qwen", "gemma"):
        if family in lowered:
            return family
    return lowered.split("/")[-1]


def meta_model_id(meta: dict) -> str:
    """
    :param meta: Parsed `meta` dict from a cache's JSON blob.
    :return: The model id field, tolerating either 'model_id' or 'model' as
        the key (extract_activations.py currently writes 'model').
    :raises KeyError: If neither key is present.
    """
    if "model_id" in meta:
        return str(meta["model_id"])
    return str(meta["model"])


@dataclass
class CacheFile:
    """
    Metadata for one activation cache, filtered to rows with usable labels.

    Feature arrays are deliberately not held here: all caches together exceed
    RAM, so `pooled_features` re-opens the npz and reads only the requested
    layer's arrays, applying `label_mask` on the fly.
    """

    path: Path
    dataset: str
    scenario: str
    base_model: str
    model_id: str
    layers: list[int]
    labels: np.ndarray
    label_mask: np.ndarray


def valid_label_mask(labels: np.ndarray) -> np.ndarray:
    """
    :param labels: Raw `deceptive` array from a cache (bool or int dtype).
    :return: Boolean mask of rows with a usable 0/1 label; integer arrays
        treat negative values (e.g. -1) as missing, bool arrays are all valid.
    """
    if np.issubdtype(labels.dtype, np.bool_):
        return np.ones(labels.shape, dtype=bool)
    return labels >= 0


def load_cache_file(path: Path) -> CacheFile | None:
    """
    Load one activation cache and validate it has usable labels.

    :param path: Path to an `.npz` cache written by extract_activations.py.
    :return: A populated CacheFile, or None if the cache has no `deceptive`
        labels at all or none of its rows have a usable label (a warning is
        printed to stdout in either case).
    """
    with np.load(path, allow_pickle=True) as data:
        if "deceptive" not in data:
            print(f"warning: {path.name} has no labels; skipping")
            return None
        raw_labels = np.asarray(data["deceptive"])
        mask = valid_label_mask(raw_labels)
        if not mask.any():
            print(f"warning: {path.name} has no usable labels (all missing/-1); skipping")
            return None
        meta = json.loads(str(data["meta"]))
        dataset = meta["dataset"]
        return CacheFile(
            path=path,
            dataset=dataset,
            scenario=parse_scenario(dataset),
            base_model=base_model_family(meta_model_id(meta)),
            model_id=meta_model_id(meta),
            layers=list(meta["layers"]),
            labels=raw_labels[mask].astype(np.int64),
            label_mask=mask,
        )


def load_cache_files(paths: Iterable[Path]) -> list[CacheFile]:
    """
    :param paths: Cache file paths to load.
    :return: Loaded caches, dropping any with no usable labels.
    """
    loaded = []
    for path in paths:
        cache = load_cache_file(path)
        if cache is not None:
            loaded.append(cache)
    return loaded


def pooled_features(cache: CacheFile, layer: int, pooling: str) -> np.ndarray:
    """
    Read one layer's pooled features from the cache's npz on demand.

    :param cache: Cache metadata (see `CacheFile` for why arrays live on disk).
    :param layer: Decoder layer index.
    :param pooling: One of 'mean', 'last', or 'concat' (mean and last
        concatenated along the feature axis).
    :return: (N, D) float32 feature matrix, cast up from the cache's float16,
        restricted to rows with usable labels. Values that overflowed float16
        during extraction (gemma's late-layer residual streams exceed 65504)
        are clipped back to the float16 range so sklearn accepts them.
    """
    keys = [f"mean_L{layer}", f"last_L{layer}"] if pooling == "concat" \
        else [f"{pooling}_L{layer}"]
    float16_max = float(np.finfo(np.float16).max)
    with np.load(cache.path, allow_pickle=True) as data:
        parts = [
            np.nan_to_num(np.asarray(data[key])[cache.label_mask].astype(np.float32),
                          nan=0.0, posinf=float16_max, neginf=-float16_max)
            for key in keys
        ]
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)


def concat_cache_features(
        cache_files: list[CacheFile], layer: int, pooling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param cache_files: Caches to concatenate, in the given order.
    :param layer: Decoder layer index.
    :param pooling: Pooling strategy, see `pooled_features`.
    :return: (features, labels, dataset_ids) with rows in cache_files order;
        dataset_ids repeats each cache's dataset id once per row.
    """
    features = np.concatenate([pooled_features(cache, layer, pooling) for cache in cache_files])
    labels = np.concatenate([cache.labels for cache in cache_files])
    dataset_ids = np.concatenate([
        np.full(len(cache.labels), cache.dataset, dtype=object) for cache in cache_files
    ])
    return features, labels, dataset_ids


def enumerate_configs(
        base_models: Iterable[str], layers: Iterable[int], poolings: Iterable[str] = POOLINGS,
) -> list[tuple[str, int, str]]:
    """
    :param base_models: Base-model family labels to sweep.
    :param layers: Layer indices to sweep.
    :param poolings: Pooling strategies to sweep.
    :return: (base_model, layer, pooling) tuples in a stable, sorted order.
    """
    return list(product(sorted(set(base_models)), sorted(set(layers)), poolings))


def has_both_classes(labels: np.ndarray) -> bool:
    """
    :param labels: Binary label array.
    :return: True if at least two distinct classes are present.
    """
    return len(np.unique(labels)) >= 2


def fit_logistic(features: np.ndarray, labels: np.ndarray, regularization: float):
    """
    :param features: (N, D) float32 training features.
    :param labels: (N,) binary training labels.
    :param regularization: Inverse regularization strength C.
    :return: A fitted scaler + logistic-regression sklearn Pipeline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, C=regularization)),
    ])
    pipeline.fit(features, labels)
    return pipeline


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """
    :param labels: True binary labels (0/1).
    :param scores: Predicted probability of the positive class.
    :return: Dict with 'auroc' (nan if only one class is present in labels)
        and 'balanced_accuracy' at a 0.5 threshold.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    predictions = (scores >= 0.5).astype(int)
    metrics = {"balanced_accuracy": balanced_accuracy_score(labels, predictions)}
    metrics["auroc"] = roc_auc_score(labels, scores) if has_both_classes(labels) else float("nan")
    return metrics


def per_dataset_metrics(
        labels: np.ndarray, scores: np.ndarray, dataset_ids: np.ndarray, **config_fields: object,
) -> list[dict]:
    """
    :param labels: True binary labels aligned with scores and dataset_ids.
    :param scores: Predicted probability of the positive class.
    :param dataset_ids: Dataset id per row.
    :param config_fields: Extra (base_model, layer, pooling, train_scenario,
        eval_scenario) fields copied into every row.
    :return: One metrics row per distinct dataset id.
    """
    rows = []
    for dataset in sorted(set(dataset_ids.tolist())):
        subset = dataset_ids == dataset
        metrics = compute_metrics(labels[subset], scores[subset])
        rows.append({
            **config_fields,
            "dataset": dataset,
            "auroc": metrics["auroc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "n_eval": int(subset.sum()),
        })
    return rows


def cross_validated_scores(
        cache_files: list[CacheFile], layer: int, pooling: str, regularization: float,
        cv_folds: int, random_state: int = 0,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """
    Out-of-fold predicted probabilities, used when no opposite-scenario data
    exists to evaluate against.

    :param cache_files: Caches for one (base_model, scenario).
    :param layer: Layer index to pool.
    :param pooling: Pooling strategy.
    :param regularization: Inverse regularization strength C.
    :param cv_folds: Requested number of stratified folds; clamped down to
        the smallest class count when that count is smaller.
    :param random_state: Fold-assignment seed.
    :return: (out_of_fold_scores, labels, dataset_ids) aligned with the
        concatenated row order of `cache_files`; scores is None if there
        are too few examples of one class to run any fold.
    """
    from sklearn.model_selection import StratifiedKFold

    features, labels, dataset_ids = concat_cache_features(cache_files, layer, pooling)
    if not has_both_classes(labels):
        return None, labels, dataset_ids
    smallest_class = min(np.bincount(labels))
    folds = min(cv_folds, smallest_class)
    if folds < 2:
        return None, labels, dataset_ids
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_scores = np.full(len(labels), np.nan)
    for train_index, test_index in splitter.split(features, labels):
        pipeline = fit_logistic(features[train_index], labels[train_index], regularization)
        oof_scores[test_index] = pipeline.predict_proba(features[test_index])[:, 1]
    return oof_scores, labels, dataset_ids


def run_sweep(
        cache_files: list[CacheFile], layers_override: list[int] | None, poolings: Iterable[str],
        regularization: float, cv_folds: int, random_state: int = 0,
) -> tuple[list[dict], list[dict]]:
    """
    Train and evaluate every (base_model, layer, pooling, train_scenario) config.

    :param cache_files: All loaded caches across every base model/scenario.
    :param layers_override: If given, restrict to these layer indices
        (intersected with the layers each base model's caches actually have).
    :param poolings: Pooling strategies to sweep.
    :param regularization: Inverse regularization strength C for LogisticRegression.
    :param cv_folds: Number of folds for the within-scenario CV fallback.
    :param random_state: Fold-assignment seed for the CV fallback.
    :return: (sweep_rows, per_dataset_rows), each a list of plain dicts.
    """
    sweep_rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    base_models = sorted({cache.base_model for cache in cache_files})
    for base_model in base_models:
        files_for_model = [cache for cache in cache_files if cache.base_model == base_model]
        scenarios_present = sorted({cache.scenario for cache in files_for_model})
        common_layers = set.intersection(*(set(cache.layers) for cache in files_for_model))
        if layers_override is not None:
            common_layers &= set(layers_override)
        layers = sorted(common_layers)
        for layer, pooling in product(layers, poolings):
            for train_scenario in scenarios_present:
                train_files = [cache for cache in files_for_model if cache.scenario == train_scenario]
                eval_scenarios = [scenario for scenario in scenarios_present if scenario != train_scenario]
                config_fields = dict(base_model=base_model, layer=layer, pooling=pooling,
                                     train_scenario=train_scenario)
                if eval_scenarios:
                    train_features, train_labels, _ = concat_cache_features(train_files, layer, pooling)
                    if not has_both_classes(train_labels):
                        print(f"warning: skipping {config_fields} (train scenario has a single class)")
                        continue
                    pipeline = fit_logistic(train_features, train_labels, regularization)
                    for eval_scenario in eval_scenarios:
                        eval_files = [cache for cache in files_for_model if cache.scenario == eval_scenario]
                        eval_features, eval_labels, eval_dataset_ids = concat_cache_features(
                            eval_files, layer, pooling)
                        scores = pipeline.predict_proba(eval_features)[:, 1]
                        metrics = compute_metrics(eval_labels, scores)
                        sweep_rows.append({
                            **config_fields, "eval_scenario": eval_scenario,
                            "auroc": metrics["auroc"], "balanced_accuracy": metrics["balanced_accuracy"],
                            "n_train": len(train_labels), "n_eval": len(eval_labels),
                        })
                        per_dataset_rows.extend(per_dataset_metrics(
                            eval_labels, scores, eval_dataset_ids,
                            **config_fields, eval_scenario=eval_scenario))
                else:
                    oof_scores, cv_labels, cv_dataset_ids = cross_validated_scores(
                        train_files, layer, pooling, regularization, cv_folds, random_state)
                    if oof_scores is None:
                        print(f"warning: skipping {config_fields} (cv fallback has a single class "
                              f"or too few examples)")
                        continue
                    metrics = compute_metrics(cv_labels, oof_scores)
                    sweep_rows.append({
                        **config_fields, "eval_scenario": CV_EVAL_LABEL,
                        "auroc": metrics["auroc"], "balanced_accuracy": metrics["balanced_accuracy"],
                        "n_train": len(cv_labels), "n_eval": len(cv_labels),
                    })
                    per_dataset_rows.extend(per_dataset_metrics(
                        cv_labels, oof_scores, cv_dataset_ids,
                        **config_fields, eval_scenario=CV_EVAL_LABEL))
    return sweep_rows, per_dataset_rows


def rows_to_frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """
    :param rows: Result rows as plain dicts.
    :param columns: Column order to enforce, used even when rows is empty.
    :return: A DataFrame with the given columns, one row per input dict.
    """
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def top_configs(sweep_frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Rank (base_model, layer, pooling) configs by cross-scenario performance.

    Averages balanced accuracy across every train/eval row sharing a config
    (both cross-scenario directions when both exist, or the single CV row
    when only one scenario is present).

    :param sweep_frame: The sweep_results rows.
    :param top_n: Number of ranked configs to keep.
    :return: Top rows sorted by mean balanced accuracy, descending.
    """
    if sweep_frame.empty:
        return pd.DataFrame(columns=["base_model", "layer", "pooling", "mean_balanced_accuracy"])
    grouped = (
        sweep_frame.groupby(["base_model", "layer", "pooling"])["balanced_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"balanced_accuracy": "mean_balanced_accuracy"})
    )
    return grouped.sort_values("mean_balanced_accuracy", ascending=False).head(top_n).reset_index(drop=True)


def parse_layers_arg(spec: str | None) -> list[int] | None:
    """
    :param spec: Comma-separated layer indices, or None.
    :return: Parsed list of ints, or None if spec is None/empty.
    """
    if not spec:
        return None
    return [int(part) for part in spec.split(",")]


def log_to_wandb(sweep_frame: pd.DataFrame, per_dataset_frame: pd.DataFrame,
                 summary: pd.DataFrame, run_config: dict, project: str, entity: str) -> None:
    """
    Log sweep results to Weights & Biases as tables plus per-config layer
    curves (balanced accuracy vs layer, one series per base_model/pooling/
    eval direction), so layer sweeps are comparable across runs.

    :param sweep_frame: The sweep_results rows.
    :param per_dataset_frame: The per-dataset breakdown rows.
    :param summary: Ranked top-configs table.
    :param run_config: CLI arguments to record on the run.
    :param project: wandb project name.
    :param entity: wandb entity (team) name.
    """
    import wandb

    run = wandb.init(project=project, entity=entity, job_type="linear-sweep",
                     config=run_config)
    run.log({
        "sweep_results": wandb.Table(dataframe=sweep_frame),
        "per_dataset": wandb.Table(dataframe=per_dataset_frame),
        "top_configs": wandb.Table(dataframe=summary),
    })
    for (base_model, pooling, train_scenario, eval_scenario), group in sweep_frame.groupby(
            ["base_model", "pooling", "train_scenario", "eval_scenario"]):
        ordered = group.sort_values("layer")
        series = f"{base_model}/{pooling}/{train_scenario}->{eval_scenario}"
        run.log({series: wandb.plot.line(
            wandb.Table(dataframe=ordered[["layer", "balanced_accuracy", "auroc"]]),
            "layer", "balanced_accuracy", title=f"balanced accuracy by layer: {series}")})
    if not summary.empty:
        best = summary.iloc[0]
        run.summary["best_mean_balanced_accuracy"] = float(best["mean_balanced_accuracy"])
        run.summary["best_config"] = f"{best['base_model']}/L{best['layer']}/{best['pooling']}"
    run.finish()


def main() -> None:
    """
    Parse CLI arguments, run the sweep, and write results to `--out-dir`.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache-dir", default="results/whitebox/activations")
    parser.add_argument("--out-dir", default="results/whitebox/linear_sweep")
    parser.add_argument("--C", dest="regularization", type=float, default=1.0,
                        help="inverse regularization strength for LogisticRegression")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="folds for the within-scenario CV fallback")
    parser.add_argument("--layers", default=None,
                        help="comma-separated layer indices to restrict the sweep to "
                             "(default: every layer common to a base model's caches)")
    parser.add_argument("--poolings", default=",".join(POOLINGS),
                        help="comma-separated pooling strategies to sweep")
    parser.add_argument("--include-limit", action="store_true",
                        help="also load '.limit' smoke caches (for pipeline smoke tests)")
    parser.add_argument("--seed", type=int, default=0, help="CV fold-assignment seed")
    parser.add_argument("--top-n", type=int, default=10, help="rows in the printed summary table")
    parser.add_argument("--wandb", action="store_true",
                        help="log results to Weights & Biases after the sweep")
    parser.add_argument("--wandb-only", action="store_true",
                        help="skip the sweep and log the CSVs already in --out-dir to wandb")
    parser.add_argument("--wandb-project", default="aletheias-quest-whitebox")
    parser.add_argument("--wandb-entity", default="aletheia-quest")
    args = parser.parse_args()

    run_config = {key: value for key, value in vars(args).items()
                  if not key.startswith("wandb")}
    out_dir = Path(args.out_dir)
    if args.wandb_only:
        sweep_frame = pd.read_csv(out_dir / "sweep_results.csv")
        per_dataset_frame = pd.read_csv(out_dir / "per_dataset.csv")
        log_to_wandb(sweep_frame, per_dataset_frame, top_configs(sweep_frame, args.top_n),
                     run_config, args.wandb_project, args.wandb_entity)
        return

    cache_dir = Path(args.cache_dir)
    paths = discover_cache_files(cache_dir, include_limit=args.include_limit)
    print(f"found {len(paths)} cache file(s) in {cache_dir} (include_limit={args.include_limit})")
    cache_files = load_cache_files(paths)
    if not cache_files:
        print("no usable cache files found; nothing to do")
        return

    poolings = args.poolings.split(",")
    layers_override = parse_layers_arg(args.layers)
    sweep_rows, per_dataset_rows = run_sweep(
        cache_files, layers_override, poolings, args.regularization, args.cv_folds, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_frame = rows_to_frame(sweep_rows, SWEEP_COLUMNS)
    per_dataset_frame = rows_to_frame(per_dataset_rows, PER_DATASET_COLUMNS)
    sweep_frame.to_csv(out_dir / "sweep_results.csv", index=False)
    per_dataset_frame.to_csv(out_dir / "per_dataset.csv", index=False)
    print(f"wrote {len(sweep_frame)} sweep rows and {len(per_dataset_frame)} per-dataset rows to {out_dir}")

    summary = top_configs(sweep_frame, args.top_n)
    if summary.empty:
        print("no configs produced results (see warnings above)")
    else:
        print(f"top {len(summary)} configs by mean cross-scenario balanced accuracy:")
        print(summary.to_string(index=False))

    if args.wandb:
        log_to_wandb(sweep_frame, per_dataset_frame, summary,
                     run_config, args.wandb_project, args.wandb_entity)


if __name__ == "__main__":
    main()
