#!/usr/bin/env python3
"""Leave-one-cell-out comparison of the v3.3 probe training recipes.

The shipped probe is selected by a *random* row split: ``train_test_split(...,
stratify=labels)`` inside ``MultiFamilyProbe.fit``.  Its early-stopping
validation rows come from the same organisms, scenarios and adapters as its
training rows, so the checkpoint chosen is the one that best fits this
scenario -- nothing in the loop ever rewards transfer.  That is the mechanism
behind AUROC 0.99 on dev and 0.7577 on Notus.

This script measures whether fixing that helps, on the only out-of-distribution
structure dev has: the (scenario x adapter) cell.

  instructed/base     1016 rows      instructed/adapter   3600 rows
  varied/base          400 rows      varied/adapter       3200 rows

For each held-out cell the probe trains on every row outside it and scores the
cell it never saw.  ``varied/base`` is the closest analogue of Notus we have --
no adapter, harder scenario -- so it is the number to watch.

Three recipes, identical except for the flagged change:

  random    the shipped recipe: random early-stopping split, uniform weights
  grouped   early stopping holds out whole organisms
  balanced  grouped, plus base-model rows upweighted to ~50% of the loss

    python experiments/nonlinear_probe/train_v3_3_probe.py --max-epochs 4   # wiring
    python experiments/nonlinear_probe/train_v3_3_probe.py                  # real run
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAMILIES = ("qwen", "gemma", "nemotron")
SEED = 0
OUT_JSON = REPO_ROOT / "results/whitebox/v3_3_cell_holdout.json"
RECIPES = ("random", "grouped", "balanced")


def subset(data: BaseModelData, rows: np.ndarray):
    """
    :param data: One family's cached rows.
    :param rows: Global row ids to repack.
    :return: ``(flat tensor on DEVICE, offsets)`` for a probe fit or score.
    """
    pieces = [data.flat[data.offsets[r]:data.offsets[r + 1]] for r in np.asarray(rows)]
    packed = np.concatenate(pieces, axis=0)
    offsets = np.cumsum([0] + [len(p) for p in pieces]).astype(np.int64)
    finfo = torch.finfo(torch.float16)
    return torch.from_numpy(packed).clamp(finfo.min, finfo.max).to(DEVICE), offsets


def cell_of(organisms: np.ndarray) -> np.ndarray:
    """
    :param organisms: ``"<scenario>/<lora or 'base'>"`` per row.
    :return: ``"<scenario>/<base|adapter>"`` per row.
    """
    return np.asarray(
        [f"{o.split('/')[0]}/{'base' if o.split('/', 1)[1] == 'base' else 'adapter'}"
         for o in organisms], dtype=object)


def base_balanced_weights(organisms: np.ndarray) -> np.ndarray:
    """
    Per-row weights that give base-model and adapter rows equal total mass.

    Base rows are 17% of the dev corpus, so the unweighted loss is dominated by
    adapter-bearing rows -- exactly the regime that carries a fine-tuning
    artefact for the probe to latch onto, and exactly the regime Notus is not.

    :param organisms: ``"<scenario>/<lora or 'base'>"`` per row.
    :return: Float weights, mean 1.0.
    """
    is_base = np.asarray([o.split("/", 1)[1] == "base" for o in organisms])
    weights = np.ones(len(organisms), dtype=np.float32)
    n_base, n_adapter = int(is_base.sum()), int((~is_base).sum())
    if n_base and n_adapter:
        weights[is_base] = 0.5 * len(organisms) / n_base
        weights[~is_base] = 0.5 * len(organisms) / n_adapter
    return weights


def fit_and_score(families: dict, train_rows: dict, test_family: str,
                  test_rows: np.ndarray, recipe: str, max_epochs: int, seed: int):
    """
    :param families: Family name to :class:`BaseModelData`.
    :param train_rows: Family name to that family's training row ids.
    :param test_family: Family whose held-out rows are scored.
    :param test_rows: Held-out row ids within ``test_family``.
    :param recipe: One of ``RECIPES``.
    :param max_epochs: Epoch cap passed to the probe.
    :param seed: Probe seed.
    :return: Deception scores for ``test_rows``.
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

    probe = MultiFamilyProbe(seed=seed, device=DEVICE, max_epochs=max_epochs).fit(
        family_data,
        family_groups=family_groups if recipe in ("grouped", "balanced") else None,
        family_weights=family_weights if recipe == "balanced" else None,
    )
    del family_data
    test_flat, test_offsets = subset(families[test_family], test_rows)
    scores = probe.predict_proba(test_family, test_flat, test_offsets)[:, 1]
    del test_flat, probe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cells", nargs="*", default=None,
                        help="held-out cells to run (default: all)")
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    manifest = load_manifest_rows()
    families = {name: BaseModelData(name, manifest) for name in FAMILIES}
    cells = {name: cell_of(data.organisms) for name, data in families.items()}

    all_cells = sorted({c for column in cells.values() for c in column.tolist()})
    wanted = args.cells or all_cells
    print(f"device={DEVICE}  cells={all_cells}")
    for cell in all_cells:
        total = sum(int((cells[name] == cell).sum()) for name in FAMILIES)
        present = [name for name in FAMILIES if (cells[name] == cell).any()]
        print(f"  {cell:<20} {total:>5} rows   families={','.join(present)}")

    results = {}
    for cell in wanted:
        results[cell] = {}
        for recipe in RECIPES:
            per_family, pooled_scores, pooled_labels = {}, [], []
            for family in FAMILIES:
                test_rows = np.flatnonzero(cells[family] == cell)
                if len(test_rows) == 0:
                    continue
                train_rows = {name: np.flatnonzero(cells[name] != cell)
                              for name in FAMILIES}
                # A family confined to a single cell (nemotron is entirely
                # instr/base) has no training rows once that cell is held out,
                # so it has no input projection to score with.  Report it as
                # not evaluable rather than silently scoring something else.
                if len(train_rows[family]) == 0:
                    per_family[family] = float("nan")
                    print(f"  {cell:<20} {recipe:<9} {family:<9} "
                          f"n={len(test_rows):>4}  SKIPPED (family confined to "
                          f"this cell; no training rows remain)", flush=True)
                    continue
                scores = fit_and_score(families, train_rows, family, test_rows,
                                       recipe, args.max_epochs, args.seed)
                labels = families[family].labels[test_rows]
                per_family[family] = (float(roc_auc_score(labels, scores))
                                      if len(np.unique(labels)) > 1 else float("nan"))
                pooled_scores.append(scores)
                pooled_labels.append(labels)
                print(f"  {cell:<20} {recipe:<9} {family:<9} "
                      f"n={len(test_rows):>4}  AUROC={per_family[family]:.4f}", flush=True)
            if not pooled_scores:
                continue
            labels = np.concatenate(pooled_labels)
            pooled = (float(roc_auc_score(labels, np.concatenate(pooled_scores)))
                      if len(np.unique(labels)) > 1 else float("nan"))
            results[cell][recipe] = {"per_family": per_family, "pooled": pooled}
            print(f"  {cell:<20} {recipe:<9} {'POOLED':<9} "
                  f"n={len(labels):>4}  AUROC={pooled:.4f}", flush=True)

    print(f"\n{'held-out cell':<22}" + "".join(f"{r:>12}" for r in RECIPES))
    for cell, by_recipe in results.items():
        row = "".join(f"{by_recipe[r]['pooled']:>12.4f}" if r in by_recipe else f"{'-':>12}"
                      for r in RECIPES)
        print(f"{cell:<22}{row}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"device": DEVICE, "seed": args.seed, "max_epochs": args.max_epochs,
         "results": results}, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
