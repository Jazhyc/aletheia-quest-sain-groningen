#!/usr/bin/env python3
"""Score one notebook against a benchmark yaml, without touching submission/ or dry.yaml.

Reuses the leaderboard's own executor + scoring, so numbers are directly
comparable to a real dry run. Purpose: measure a candidate Notus detector
(behavioral, logit-readout, ...) on the labeled dev base-model ("None") sets --
our offline proxy for Notus -- before spending a submission on it.

Usage:
    NDIF_API_KEY=...  HF_TOKEN=...  python experiments/behavioral_notus/run.py \\
        --notebook baselines/unrelated_questions_probe.ipynb \\
        --yaml experiments/behavioral_notus/dry.none.yaml

The notebook is run once per dataset with DATASET_NAME set, exactly as the real
runner does; its submission.csv is scored against the dataset's labels.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "leaderboard" / "src"))

import yaml  # noqa: E402
from aletheia_runner import executor, scoring  # noqa: E402
from aletheia_runner.config import DatasetConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True,
                        help="Notebook to score (path relative to repo root or absolute).")
    parser.add_argument("--yaml", default="experiments/behavioral_notus/dry.none.yaml",
                        help="Benchmark yaml listing datasets + labels_uri.")
    parser.add_argument("--limit", type=int, default=None,
                        help="ALETHEIA_LIMIT: score only the first N rows per dataset (fast smoke test).")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-dataset wall-clock budget in seconds (matches the real runner).")
    parser.add_argument("--ndif-api-key", default=os.environ.get("NDIF_API_KEY"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def load_datasets(yaml_path: Path) -> list[DatasetConfig]:
    data = yaml.safe_load(yaml_path.read_text())
    return [DatasetConfig(**d) for d in data.get("datasets", [])]


def main() -> int:
    args = parse_args()
    nb_path = Path(args.notebook)
    if not nb_path.is_absolute():
        nb_path = REPO_ROOT / nb_path
    if not nb_path.exists():
        sys.exit(f"notebook not found: {nb_path}")

    datasets = load_datasets((REPO_ROOT / args.yaml) if not Path(args.yaml).is_absolute() else Path(args.yaml))
    if not datasets:
        sys.exit(f"no datasets in {args.yaml}")

    base_env = {"NNSIGHT_REMOTE": "1"}
    if args.ndif_api_key:
        base_env["NDIF_API_KEY"] = args.ndif_api_key
    if args.hf_token:
        base_env["HF_TOKEN"] = args.hf_token
    if args.limit is not None:
        base_env["ALETHEIA_LIMIT"] = str(args.limit)

    snapshot_dir = REPO_ROOT / "results" / "behavioral_notus" / nb_path.stem
    print(f"notebook : {nb_path.relative_to(REPO_ROOT)}")
    print(f"benchmark: {args.yaml}  ({len(datasets)} datasets)")
    print(f"snapshots: {snapshot_dir.relative_to(REPO_ROOT)}")
    print("-" * 92)

    labels_cache: dict[str, object] = {}
    aurocs, baccs = [], []
    for ds in datasets:
        env = {**base_env, "DATASET_NAME": ds.name}
        res = executor.run_notebook(nb_path, REPO_ROOT, env, args.timeout, snapshot_dir)
        short = ds.name.split("/")[-1]
        if not res.ok:
            print(f"  FAIL  {short:70s}  {res.error}")
            continue
        try:
            if ds.key not in labels_cache:
                labels_cache[ds.key] = scoring.load_labels(ds, args.hf_token)
            preds = scoring.load_predictions(res.submission_csv)
            m = scoring.compute_metrics(preds, labels_cache[ds.key])
        except Exception as exc:  # scoring/label errors shouldn't abort the sweep
            print(f"  SCOREFAIL  {short:64s}  {type(exc).__name__}: {exc}")
            continue
        auroc, bacc = m.get("auroc"), m.get("balanced_accuracy")
        rec, fpr = m.get("recall"), m.get("fpr")
        if auroc is not None:
            aurocs.append(auroc)
        if bacc is not None:
            baccs.append(bacc)
        print(f"  ok    {short:70s}  AUROC {auroc:.4f}  BA {bacc:.4f}  "
              f"recall {rec:.4f}  FPR {fpr:.4f}")

    print("-" * 92)
    if aurocs:
        print(f"  MEAN over {len(aurocs)} scored: AUROC {sum(aurocs)/len(aurocs):.4f}  "
              f"BA {sum(baccs)/len(baccs):.4f}")
        print("  (Notus proxy bar to beat: our probe's real Notus AUROC is 0.69 / 0.76 / 0.83)")
    else:
        print("  no datasets scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
