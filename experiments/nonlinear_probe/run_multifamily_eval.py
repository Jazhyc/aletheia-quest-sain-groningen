"""Does pooling all base-model families into one shared-trunk probe beat the
shipped one-probe-per-family design?

Two experiments, both leakage-free and both on cached activations (no NDIF):

  loo       -- leave-one-organism-out over the qwen and gemma organisms. The
               single-family probe trains on that family's other organisms; the
               shared probe trains on every other organism from EVERY family.
               Tests whether pooling helps a family that already has data.

  nemotron  -- 5-fold row cross-validation over the 216 Nemotron rows, pooling
               out-of-fold predictions into one AUROC. This is the experiment
               that matters: Nemotron's entire corpus is 216 rows with 16
               positives, and Notus/Nemotron is 52% of our AUROC gap to rank 1.
               The single-family probe sees only the other Nemotron rows; the
               shared probe additionally sees all qwen and gemma rows.

    python experiments/nonlinear_probe/run_multifamily_eval.py --experiments nemotron
    python experiments/nonlinear_probe/run_multifamily_eval.py --max-epochs 4 \
        --limit-organisms 2                      # fast wiring check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments/ensemble_gate_eval"))
sys.path.insert(0, str(HERE))

from multifamily_probe import MultiFamilyProbe  # noqa: E402
from run_eval import BaseModelData, load_manifest_rows  # noqa: E402
from token_probes import TokenProbe  # noqa: E402

OUT_JSON = REPO_ROOT / "results/whitebox/multifamily_eval.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAMILIES = ("qwen", "gemma", "nemotron")


def subset(data: BaseModelData, rows: np.ndarray):
    """Repack rows of one family into (flat tensor on DEVICE, offsets)."""
    pieces = [data.flat[data.offsets[r]:data.offsets[r + 1]] for r in np.asarray(rows)]
    packed = np.concatenate(pieces, axis=0)
    offsets = np.cumsum([0] + [len(p) for p in pieces]).astype(np.int64)
    finfo = torch.finfo(torch.float16)
    return torch.from_numpy(packed).clamp(finfo.min, finfo.max).to(DEVICE), offsets


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def score_single_family(data: BaseModelData, train_rows, test_rows, max_epochs, seed):
    """Fit the shipped design: one probe on one family's train rows."""
    train_flat, train_offsets = subset(data, train_rows)
    probe = TokenProbe("transformer", seed=seed, device=DEVICE,
                       max_epochs=max_epochs).fit(train_flat, train_offsets,
                                                  data.labels[train_rows])
    del train_flat
    test_flat, test_offsets = subset(data, test_rows)
    scores = probe.predict_proba(test_flat, test_offsets)[:, 1]
    del test_flat, probe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return scores


def score_shared_trunk(families: dict, train_rows_by_family: dict, test_family: str,
                       test_rows, max_epochs, seed):
    """Fit one shared-trunk probe on every family's train rows, score one family."""
    family_data = {}
    for family, rows in train_rows_by_family.items():
        if len(rows) == 0:
            continue
        flat, offsets = subset(families[family], rows)
        family_data[family] = (flat, offsets, families[family].labels[rows])
    probe = MultiFamilyProbe(seed=seed, device=DEVICE, max_epochs=max_epochs).fit(family_data)
    del family_data
    test_flat, test_offsets = subset(families[test_family], test_rows)
    scores = probe.predict_proba(test_family, test_flat, test_offsets)[:, 1]
    del test_flat, probe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return scores


def run_loo(families: dict, max_epochs: int, seed: int, limit: int | None) -> dict:
    results = {}
    for family in ("qwen", "gemma"):
        if family not in families:
            continue
        data = families[family]
        organisms = sorted(set(data.organisms.tolist()))
        if limit:
            organisms = organisms[:limit]
        if len(organisms) < 2:
            continue
        all_rows = np.arange(len(data.labels))
        for held in organisms:
            test_rows = all_rows[data.organisms == held]
            train_rows = all_rows[data.organisms != held]
            if len(np.unique(data.labels[train_rows])) < 2:
                continue
            test_y = data.labels[test_rows]
            single = score_single_family(data, train_rows, test_rows, max_epochs, seed)
            train_by_family = {other: (train_rows if other == family
                                       else np.arange(len(families[other].labels)))
                               for other in families}
            shared = score_shared_trunk(families, train_by_family, family,
                                        test_rows, max_epochs, seed)
            results[f"{family}:{held}"] = {
                "n": int(len(test_y)),
                "single_family": auroc(test_y, single),
                "shared_trunk": auroc(test_y, shared),
            }
            row = results[f"{family}:{held}"]
            print(f"  {family}:{held:<26} n={row['n']:<5} "
                  f"single={row['single_family']:.4f}  shared={row['shared_trunk']:.4f}",
                  flush=True)
    return results


def run_nemotron(families: dict, max_epochs: int, seed: int, n_splits: int = 5) -> dict:
    """5-fold row CV over Nemotron, pooling out-of-fold scores into one AUROC.

    216 rows with 16 positives is far too few for a single holdout, so every row
    is scored exactly once out-of-fold and the AUROC is computed on the pooled
    predictions.
    """
    data = families["nemotron"]
    labels = data.labels
    single_scores = np.zeros(len(labels))
    shared_scores = np.zeros(len(labels))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_rows, test_rows) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        single_scores[test_rows] = score_single_family(
            data, train_rows, test_rows, max_epochs, seed)
        train_by_family = {family: (train_rows if family == "nemotron"
                                    else np.arange(len(families[family].labels)))
                           for family in families}
        shared_scores[test_rows] = score_shared_trunk(
            families, train_by_family, "nemotron", test_rows, max_epochs, seed)
        print(f"  nemotron fold {fold + 1}/{n_splits}: {len(test_rows)} test rows "
              f"({int(labels[test_rows].sum())} positive)", flush=True)
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "single_family": auroc(labels, single_scores),
        "shared_trunk": auroc(labels, shared_scores),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=["nemotron", "loo"],
                        choices=["nemotron", "loo"])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-organisms", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    print(f"device = {DEVICE}  seed = {args.seed}  max_epochs = {args.max_epochs}")
    manifest = load_manifest_rows()
    families = {}
    for family in FAMILIES:
        data = BaseModelData(family, manifest)
        families[family] = data
        print(f"  {family}: {len(data.labels)} rows, "
              f"{len(set(data.organisms.tolist()))} organisms, "
              f"{int(data.labels.sum())} positive, hidden {data.flat.shape[1]}", flush=True)

    report = {}
    if "nemotron" in args.experiments:
        print("\n### Nemotron 5-fold row CV (the 52%-of-the-gap cell)")
        report["nemotron"] = run_nemotron(families, args.max_epochs, args.seed)
        result = report["nemotron"]
        print(f"\n  pooled out-of-fold AUROC over {result['n']} rows "
              f"({result['n_positive']} positive):")
        print(f"    single_family  {result['single_family']:.4f}")
        print(f"    shared_trunk   {result['shared_trunk']:.4f}"
              f"   ({result['shared_trunk'] - result['single_family']:+.4f})")

    if "loo" in args.experiments:
        print("\n### Leave-one-organism-out (qwen + gemma)")
        per_organism = run_loo(families, args.max_epochs, args.seed, args.limit_organisms)
        report["loo"] = per_organism
        if per_organism:
            print(f"\n{'method':<16}{'mean AUROC':>12}{'WORST AUROC':>14}")
            for method in ("single_family", "shared_trunk"):
                values = [row[method] for row in per_organism.values()
                          if not np.isnan(row[method])]
                print(f"{method:<16}{np.mean(values):>12.4f}{np.min(values):>14.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
