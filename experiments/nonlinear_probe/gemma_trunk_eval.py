#!/usr/bin/env python3
"""Is the shared trunk better than gemma's single-family probe?

`sonic_v2.3.7` moved qwen and nemotron onto the shared multi-family trunk but
left **gemma on its old single-family probe**, and no measurement in the repo
says why. Shipping gemma's trunk head is an export change, not a retrain, so
this is worth resolving.

Three leakage-free holdouts, each training both designs on identical rows:

  instr/base      hold out gemma's base-model cell (400 rows)
  instr/adapter   hold out gemma's adapter cell (400 rows)
  g-st            hold out `dev-instructed-deception-gemma-3-27b-it-g-st-...`

The third is the interesting one. Those 400 rows are downloaded and extracted
but were never added to `dev_splits/manifest.csv`, so no shipped probe has ever
trained on them -- and `g-st` is the organism family of all three counted Iris
units. It is the only leakage-free proxy we have for a counted dataset.

  single   TokenProbe("transformer") on gemma's training rows alone
  shared   MultiFamilyProbe on every family's training rows

Both use the `balanced` recipe (grouped early-stopping split + base-row
upweighting) where the design supports it; TokenProbe has no grouped-split
parameter, so `single` keeps its random split -- that asymmetry is the shipped
state and is noted in the output.

NOTE ON NOISE: repeated runs of the same configuration differ by ~0.0012 from
GPU non-determinism, so treat single-cell gaps below ~0.002 as ties.

    python experiments/nonlinear_probe/gemma_trunk_eval.py --max-epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments/ensemble_gate_eval"))
sys.path.insert(0, str(HERE))

from multifamily_probe import MultiFamilyProbe  # noqa: E402
from run_eval import BaseModelData, load_manifest_rows  # noqa: E402
from token_probes import TokenProbe  # noqa: E402
from train_v3_3_probe import base_balanced_weights, cell_of, subset  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAMILIES = ("qwen", "gemma", "nemotron")
OUT_JSON = REPO_ROOT / "results/whitebox/gemma_trunk_eval.json"
GST_CACHE = (REPO_ROOT / "results/whitebox/activations/"
             "aletheias-quest__dev-instructed-deception-gemma-3-27b-it-"
             "g-st-gemma-3-27b-it-2.tokens.npz")


def load_gst(layer: int):
    """
    Load the held-out `g-st` gemma rows straight from the activation cache.

    These rows are deliberately absent from the training manifest, so they are
    loaded here rather than through :class:`BaseModelData`.

    :param layer: Residual layer to read.
    :return: ``(flat tensor on DEVICE, offsets, labels)``.
    """
    with np.load(GST_CACHE, allow_pickle=True) as data:
        tokens = data[f"tokens_L{layer}"]
        offsets = data["token_offsets"].astype(np.int64)
        labels = data["deceptive"].astype(int)
    finfo = torch.finfo(torch.float16)
    flat = torch.from_numpy(tokens).clamp(finfo.min, finfo.max).to(DEVICE)
    return flat, offsets, labels


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    :param labels: Binary labels.
    :param scores: Continuous scores.
    :return: AUROC, or NaN when the labels are single-class.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fit_single(gemma: BaseModelData, train_rows: np.ndarray, max_epochs: int, seed: int):
    """
    Fit the shipped single-family design on gemma's rows alone.

    :param gemma: Gemma's cached rows.
    :param train_rows: Gemma row ids to train on.
    :param max_epochs: Epoch cap.
    :param seed: Probe seed.
    :return: Fitted :class:`TokenProbe`.
    """
    flat, offsets = subset(gemma, train_rows)
    return TokenProbe("transformer", seed=seed, device=DEVICE,
                      max_epochs=max_epochs).fit(flat, offsets, gemma.labels[train_rows])


def fit_shared(families: dict, train_rows: dict, max_epochs: int, seed: int):
    """
    Fit the shared trunk on every family's training rows, balanced recipe.

    :param families: Family name to :class:`BaseModelData`.
    :param train_rows: Family name to that family's training row ids.
    :param max_epochs: Epoch cap.
    :param seed: Probe seed.
    :return: Fitted :class:`MultiFamilyProbe`.
    """
    family_data, family_groups, family_weights = {}, {}, {}
    for family, rows in train_rows.items():
        if len(rows) == 0:
            continue
        flat, offsets = subset(families[family], rows)
        family_data[family] = (flat, offsets, families[family].labels[rows])
        organisms = families[family].organisms[rows]
        family_groups[family] = organisms
        family_weights[family] = base_balanced_weights(organisms)
    return MultiFamilyProbe(seed=seed, device=DEVICE, max_epochs=max_epochs).fit(
        family_data, family_groups=family_groups, family_weights=family_weights)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=46)
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    if args.layer != 46:
        import run_eval
        run_eval.probe_layer = lambda _model, _layer=args.layer: _layer

    manifest = load_manifest_rows()
    families = {name: BaseModelData(name, manifest) for name in FAMILIES}
    cells = {name: cell_of(data.organisms) for name, data in families.items()}
    gemma = families["gemma"]
    results = {}

    print(f"device={DEVICE}  layer={args.layer}  gemma rows={len(gemma.labels)}")

    for cell in sorted(set(cells["gemma"].tolist())):
        test_rows = np.flatnonzero(cells["gemma"] == cell)
        train_rows = {name: np.flatnonzero(cells[name] != cell) for name in FAMILIES}
        labels = gemma.labels[test_rows]

        probe = fit_single(gemma, train_rows["gemma"], args.max_epochs, args.seed)
        flat, offsets = subset(gemma, test_rows)
        single = auroc(labels, probe.predict_proba(flat, offsets)[:, 1])
        del probe, flat

        trunk = fit_shared(families, train_rows, args.max_epochs, args.seed)
        flat, offsets = subset(gemma, test_rows)
        shared = auroc(labels, trunk.predict_proba("gemma", flat, offsets)[:, 1])
        del trunk, flat
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        results[cell] = {"single": single, "shared": shared, "n": int(len(test_rows))}
        print(f"  {cell:<16} n={len(test_rows):>4}  single={single:.4f}  "
              f"shared={shared:.4f}  delta={shared - single:+.4f}", flush=True)

    if GST_CACHE.exists():
        flat, offsets, labels = load_gst(args.layer)
        all_rows = {name: np.arange(len(families[name].labels)) for name in FAMILIES}

        probe = fit_single(gemma, all_rows["gemma"], args.max_epochs, args.seed)
        single = auroc(labels, probe.predict_proba(flat, offsets)[:, 1])
        del probe

        trunk = fit_shared(families, all_rows, args.max_epochs, args.seed)
        shared = auroc(labels, trunk.predict_proba("gemma", flat, offsets)[:, 1])
        del trunk
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        results["g-st"] = {"single": single, "shared": shared, "n": int(len(labels))}
        print(f"  {'g-st (unseen)':<16} n={len(labels):>4}  single={single:.4f}  "
              f"shared={shared:.4f}  delta={shared - single:+.4f}", flush=True)
    else:
        print(f"[WARN] {GST_CACHE} missing; skipping the g-st holdout", file=sys.stderr)

    print(f"\n{'holdout':<18}{'single':>10}{'shared':>10}{'delta':>10}")
    for name, row in results.items():
        print(f"{name:<18}{row['single']:>10.4f}{row['shared']:>10.4f}"
              f"{row['shared'] - row['single']:>+10.4f}")
    print("\nGaps below ~0.002 are within run-to-run GPU noise.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"device": DEVICE, "layer": args.layer, "seed": args.seed,
         "max_epochs": args.max_epochs, "results": results}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
