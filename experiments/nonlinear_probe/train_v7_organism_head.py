#!/usr/bin/env python3
"""Phase 2 comparison: frozen encoder + LR head vs joint two-head training.

Compares two approaches for adding an organism-detection head to the L46 probe,
using non-leaked leave-one-cell-out evaluation:

  Option A: freeze existing v3.3 encoder, train only Linear(128→1) on
            pooled features → is_organism (logistic regression).

  Option B: train a fresh two-head TransformerTokenProbe from scratch with
            joint loss L = BCE_deception + λ·BCE_organism.

Holds out one (scenario × base/adapter) cell at a time, trains on all others,
scores the held-out cell, and aggregates across all 4 cell types.

Run:
    python experiments/nonlinear_probe/train_v7_organism_head.py --max-epochs 4   # smoke
    python experiments/nonlinear_probe/train_v7_organism_head.py                  # full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments/ensemble_gate_eval"))
sys.path.insert(0, str(HERE))

from multifamily_probe import MultiFamilyProbe, MultiFamilyTokenProbe  # noqa: E402
from run_eval import BaseModelData, load_manifest_rows  # noqa: E402
from token_probes import (pack_length_sorted_batches,  # noqa: E402
                          sinusoidal_position_encoding)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAMILIES = ("qwen", "gemma", "nemotron")
SEED = 42
OUT_JSON = REPO_ROOT / "results/whitebox/v7_organism_head_comparison.json"

# The 4 cell types for leave-one-out evaluation
CELLS = ("instr/base", "instr/adapter", "varied/base", "varied/adapter")


# ---------------------------------------------------------------------------
# Data helpers (reused from train_v3_3_probe.py)
# ---------------------------------------------------------------------------

def subset(data: BaseModelData, rows: np.ndarray):
    pieces = [data.flat[data.offsets[r]:data.offsets[r + 1]] for r in np.asarray(rows)]
    packed = np.concatenate(pieces, axis=0)
    offsets = np.cumsum([0] + [len(p) for p in pieces]).astype(np.int64)
    finfo = torch.finfo(torch.float16)
    return torch.from_numpy(packed).clamp(finfo.min, finfo.max).to(DEVICE), offsets


def cell_of(organisms: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{o.split('/')[0]}/{'base' if o.split('/', 1)[1] == 'base' else 'adapter'}"
         for o in organisms], dtype=object)


def is_organism_label(organisms: np.ndarray) -> np.ndarray:
    """1 if lora is not 'base', 0 otherwise."""
    return np.asarray(
        [0 if o.split("/", 1)[1] == "base" else 1 for o in organisms],
        dtype=np.int64)


# ---------------------------------------------------------------------------
# Option A: frozen encoder + logistic regression organism head
# ---------------------------------------------------------------------------

def train_option_a(families: dict, train_rows: dict,
                   test_family: str, test_rows: np.ndarray,
                   max_epochs: int):
    """
    1. Train a standard MultiFamilyProbe (deception only) on the training split.
    2. Extract pooled features for all train rows.
    3. Train LogisticRegression on pooled → is_organism.
    4. Score deception + organism on held-out test rows.

    Returns: (deception_scores, organism_logits, deception_labels, organism_labels)
    """
    # --- Step 1: Train deception probe on train split ---
    family_data, family_groups = {}, {}
    for family, rows in train_rows.items():
        if len(rows) == 0:
            continue
        flat, offsets = subset(families[family], rows)
        family_data[family] = (flat, offsets, families[family].labels[rows])
        family_groups[family] = families[family].organisms[rows]

    probe = MultiFamilyProbe(seed=SEED, device=DEVICE, max_epochs=max_epochs).fit(
        family_data, family_groups=family_groups)
    model: MultiFamilyTokenProbe = probe.model

    # --- Step 2: Extract pooled features for train rows ---
    train_pooled = []
    train_org_labels = []
    for family, rows in train_rows.items():
        if len(rows) == 0:
            continue
        flat, offsets = subset(families[family], rows)
        org_labels = is_organism_label(families[family].organisms[rows])
        pooled = _extract_pooled(model, family, flat, offsets, probe.moments)
        train_pooled.append(pooled.cpu().numpy())
        train_org_labels.append(org_labels)

    X_train = np.concatenate(train_pooled, axis=0)
    y_train = np.concatenate(train_org_labels, axis=0)

    # --- Step 3: Train logistic regression ---
    if len(np.unique(y_train)) < 2:
        # Only one class in training — organism head can't train
        org_head = None
    else:
        org_head = LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=SEED)
        org_head.fit(X_train, y_train)

    # --- Step 4: Score held-out test rows ---
    test_flat, test_offsets = subset(families[test_family], test_rows)
    deception_scores = probe.predict_proba(test_family, test_flat, test_offsets)[:, 1]

    if org_head is not None:
        test_pooled = _extract_pooled(model, test_family, test_flat,
                                       test_offsets, probe.moments).cpu().numpy()
        organism_logits = org_head.decision_function(test_pooled)
    else:
        organism_logits = np.zeros(len(test_rows), dtype=np.float64)

    deception_labels = families[test_family].labels[test_rows]
    organism_labels = is_organism_label(families[test_family].organisms[test_rows])

    del model, probe, test_flat
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return deception_scores, organism_logits, deception_labels, organism_labels


def _extract_pooled(model: MultiFamilyTokenProbe, family: str,
                    flat_features: torch.Tensor, offsets: np.ndarray,
                    moments: dict) -> torch.Tensor:
    """Forward-pass through encoder only, return pooled features before head."""
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    num_rows = len(offsets_arr) - 1
    lengths = (offsets_arr[1:] - offsets_arr[:-1]).tolist()

    mean, std = moments[family]
    all_pooled = []

    model.eval()
    with torch.no_grad():
        for row_ids in pack_length_sorted_batches(lengths, 8192):
            # Build batch (same as MultiFamilyProbe._build_batch)
            max_len = max(lengths[r] for r in row_ids)
            padded = torch.zeros((len(row_ids), max_len, flat_features.shape[1]),
                                 dtype=torch.float32, device=flat_features.device)
            mask = torch.zeros((len(row_ids), max_len), dtype=torch.bool,
                               device=flat_features.device)
            for pos, row in enumerate(row_ids):
                s, e = int(offsets_arr[row]), int(offsets_arr[row + 1])
                padded[pos, :e - s] = flat_features[s:e].to(torch.float32)
                mask[pos, :e - s] = True

            x = (padded - mean) / std * mask.unsqueeze(-1)
            seq_len = x.shape[1]
            pe = sinusoidal_position_encoding(seq_len, model.d_model, device=x.device)
            tokens = model.projections[family](x) + pe.unsqueeze(0)
            encoded = model.encoder(tokens, src_key_padding_mask=~mask)
            m = mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            all_pooled.append(pooled)

    return torch.cat(all_pooled, dim=0)


# ---------------------------------------------------------------------------
# Option B: joint two-head training from scratch
# ---------------------------------------------------------------------------

class TwoHeadTokenProbe(nn.Module):
    """MultiFamilyTokenProbe with two heads: deception + organism."""

    def __init__(self, hidden_dims: dict[str, int], d_model: int = 128,
                 n_heads: int = 4, dim_feedforward: int = 256,
                 n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.projections = nn.ModuleDict(
            {family: nn.Linear(width, d_model) for family, width in hidden_dims.items()})
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.head_deception = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.head_organism = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, family: str, padded_tokens: torch.Tensor,
                padding_mask: torch.Tensor):
        seq_len = padded_tokens.shape[1]
        pe = sinusoidal_position_encoding(seq_len, self.d_model, device=padded_tokens.device)
        tokens = self.projections[family](padded_tokens) + pe.unsqueeze(0)
        encoded = self.encoder(tokens, src_key_padding_mask=~padding_mask)
        m = padding_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return (self.head_deception(pooled).squeeze(-1),
                self.head_organism(pooled).squeeze(-1))


def train_option_b(families: dict, train_rows: dict,
                   test_family: str, test_rows: np.ndarray,
                   max_epochs: int, organism_loss_weight: float = 1.0):
    """
    Train TwoHeadTokenProbe from scratch with joint loss.
    Uses the same training loop as MultiFamilyProbe but with two heads.
    """
    # --- Prepare training data ---
    family_data = {}
    for family, rows in train_rows.items():
        if len(rows) == 0:
            continue
        flat, offsets = subset(families[family], rows)
        family_data[family] = (flat, offsets,
                               families[family].labels[rows],
                               is_organism_label(families[family].organisms[rows]))

    # Compute per-family moments for standardisation
    moments = {}
    for family, (flat, offsets, _, _) in family_data.items():
        from token_probes import streaming_token_moments
        offsets_arr = np.asarray(offsets, dtype=np.int64)
        mean, std = streaming_token_moments(flat, offsets_arr,
                                            np.arange(len(offsets_arr) - 1))
        moments[family] = (mean.to(DEVICE), std.to(DEVICE))

    # Build model
    hidden_dims = {family: int(data[0].shape[1]) for family, data in family_data.items()}
    model = TwoHeadTokenProbe(hidden_dims).to(DEVICE)

    # Prepare batches
    train_batches = []
    for family, (flat, offsets, deception_labels, organism_labels) in family_data.items():
        offsets_arr = np.asarray(offsets, dtype=np.int64)
        lengths = offsets_arr[1:] - offsets_arr[:-1]
        n_rows = len(lengths)
        for batch in pack_length_sorted_batches(lengths.tolist(), 8192):
            train_batches.append((family, [int(p) for p in batch],
                                  torch.from_numpy(deception_labels.astype(np.float32)),
                                  torch.from_numpy(organism_labels.astype(np.float32))))

    # Simple 85/15 train/val split for early stopping (same as MultiFamilyProbe)
    generator = np.random.default_rng(SEED)
    perm = generator.permutation(len(train_batches))
    n_val = max(1, int(len(train_batches) * 0.15))
    val_idx = set(perm[:n_val].tolist())
    train_idx = [int(i) for i in perm[n_val:]]

    # Losses
    total_pos = sum(int(org.sum()) for _, _, _, org in train_batches)
    total_rows = sum(len(org) for _, _, _, org in train_batches)
    pos_weight = torch.tensor(
        [(total_rows - total_pos) / max(1, total_pos)], device=DEVICE)

    deception_loss_fn = nn.BCEWithLogitsLoss()
    organism_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    patience = 12
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        batch_order = generator.permutation(train_idx)
        for bi in batch_order:
            family, row_ids, dec_labels, org_labels = train_batches[bi]
            flat, offsets, _, _ = family_data[family]
            batch_features, batch_mask = _build_batch(
                family, flat, np.asarray(offsets, dtype=np.int64),
                row_ids, moments)
            optimizer.zero_grad()
            dec_logits, org_logits = model(family, batch_features, batch_mask)
            d_loss = deception_loss_fn(dec_logits, dec_labels[row_ids].to(DEVICE))
            o_loss = organism_loss_fn(org_logits, org_labels[row_ids].to(DEVICE))
            loss = d_loss + organism_loss_weight * o_loss
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses, n_val_total = [], 0
        with torch.no_grad():
            for bi in val_idx:
                family, row_ids, dec_labels, org_labels = train_batches[bi]
                flat, offsets, _, _ = family_data[family]
                batch_features, batch_mask = _build_batch(
                    family, flat, np.asarray(offsets, dtype=np.int64),
                    row_ids, moments)
                dec_logits, org_logits = model(family, batch_features, batch_mask)
                d_loss = deception_loss_fn(dec_logits, dec_labels[row_ids].to(DEVICE))
                o_loss = organism_loss_fn(org_logits, org_labels[row_ids].to(DEVICE))
                vl = (d_loss + organism_loss_weight * o_loss).item()
                val_losses.append(vl * len(row_ids))
                n_val_total += len(row_ids)

        val_loss = sum(val_losses) / n_val_total
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Score held-out test rows ---
    test_flat, test_offsets = subset(families[test_family], test_rows)
    offsets_arr = np.asarray(test_offsets, dtype=np.int64)
    num_rows = len(offsets_arr) - 1
    lengths = (offsets_arr[1:] - offsets_arr[:-1]).tolist()

    dec_scores = np.zeros(num_rows, dtype=np.float64)
    org_logits = np.zeros(num_rows, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for row_ids in pack_length_sorted_batches(lengths, 8192):
            batch_features, batch_mask = _build_batch(
                test_family, test_flat, offsets_arr, row_ids, moments)
            d_logits, o_logits = model(test_family, batch_features, batch_mask)
            d_probs = torch.sigmoid(d_logits).cpu().numpy()
            o_raw = o_logits.cpu().numpy()
            dec_scores[row_ids] = d_probs
            org_logits[row_ids] = o_raw

    deception_labels = families[test_family].labels[test_rows]
    organism_labels = is_organism_label(families[test_family].organisms[test_rows])

    del model, test_flat
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return dec_scores, org_logits, deception_labels, organism_labels


def _build_batch(family: str, flat_features: torch.Tensor, offsets: np.ndarray,
                 row_ids: list[int], moments: dict):
    mean, std = moments[family]
    lengths = [int(offsets[row + 1] - offsets[row]) for row in row_ids]
    max_len = max(lengths)
    padded = torch.zeros((len(row_ids), max_len, flat_features.shape[1]),
                         dtype=torch.float32, device=flat_features.device)
    mask = torch.zeros((len(row_ids), max_len), dtype=torch.bool,
                       device=flat_features.device)
    for pos, row in enumerate(row_ids):
        s, e = int(offsets[row]), int(offsets[row + 1])
        padded[pos, :e - s] = flat_features[s:e].to(torch.float32)
        mask[pos, :e - s] = True
    return (padded - mean) / std * mask.unsqueeze(-1), mask


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def auroc_or_nan(scores, labels):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-epochs", type=int, default=60,
                        help="max epochs for probe training (default: 60)")
    parser.add_argument("--cells", nargs="*", default=None,
                        help="held-out cells to evaluate (default: all 4)")
    parser.add_argument("--options", nargs="*", default=["A", "B"],
                        help="which options to run: A, B, or both")
    parser.add_argument("--organism-loss-weight", type=float, default=1.0,
                        help="λ for organism BCE in Option B joint loss")
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    # Load all data
    manifest = load_manifest_rows()
    families = {name: BaseModelData(name, manifest) for name in FAMILIES}
    cells = {name: cell_of(data.organisms) for name, data in families.items()}

    all_cells = sorted({c for col in cells.values() for c in col.tolist()})
    wanted = args.cells or [c for c in CELLS if c in all_cells]

    print(f"device={DEVICE}  max_epochs={args.max_epochs}  options={args.options}")
    print(f"cells: {all_cells}")
    for name, data in families.items():
        print(f"  {name}: {len(data.labels)} rows, "
              f"organisms={len(np.unique(data.organisms))}, "
              f"avg_len={data.flat.shape[0] / len(data.labels):.0f} tokens")

    results = {}
    for cell in wanted:
        results[cell] = {}
        for option in args.options:
            print(f"\n{'='*60}")
            print(f"CELL={cell}  OPTION={option}")
            print(f"{'='*60}")

            per_family = {}
            pooled_dec, pooled_org_logits, pooled_dec_labels, pooled_org_labels = [], [], [], []

            for family in FAMILIES:
                test_rows = np.flatnonzero(cells[family] == cell)
                if len(test_rows) == 0:
                    continue
                train_rows = {name: np.flatnonzero(cells[name] != cell)
                              for name in FAMILIES}
                if len(train_rows[family]) == 0:
                    print(f"  {family}: SKIPPED (no training rows)")
                    continue

                if option == "A":
                    dec_scores, org_logits, dec_labels, org_labels = train_option_a(
                        families, train_rows, family, test_rows, args.max_epochs)
                else:  # B
                    dec_scores, org_logits, dec_labels, org_labels = train_option_b(
                        families, train_rows, family, test_rows,
                        args.max_epochs, args.organism_loss_weight)

                dec_auc = auroc_or_nan(dec_scores, dec_labels)
                org_auc = auroc_or_nan(org_logits, org_labels)
                pearson_r = float(np.corrcoef(dec_scores, org_logits)[0, 1]) if len(dec_scores) > 2 else float("nan")

                per_family[family] = {
                    "n": len(test_rows),
                    "deception_auroc": dec_auc,
                    "organism_auroc": org_auc,
                    "pearson_r_deception_organism": pearson_r,
                }
                pooled_dec.append(dec_scores)
                pooled_org_logits.append(org_logits)
                pooled_dec_labels.append(dec_labels)
                pooled_org_labels.append(org_labels)

                print(f"  {family:10s} n={len(test_rows):>4}  "
                      f"dec_auc={dec_auc:.4f}  org_auc={org_auc:.4f}  "
                      f"r={pearson_r:+.3f}")

            if not pooled_dec:
                continue

            all_dec = np.concatenate(pooled_dec)
            all_org = np.concatenate(pooled_org_logits)
            all_dec_l = np.concatenate(pooled_dec_labels)
            all_org_l = np.concatenate(pooled_org_labels)

            pooled_dec_auc = auroc_or_nan(all_dec, all_dec_l)
            pooled_org_auc = auroc_or_nan(all_org, all_org_l)
            pooled_r = float(np.corrcoef(all_dec, all_org)[0, 1]) if len(all_dec) > 2 else float("nan")

            results[cell][option] = {
                "per_family": per_family,
                "pooled": {
                    "n": len(all_dec),
                    "deception_auroc": pooled_dec_auc,
                    "organism_auroc": pooled_org_auc,
                    "pearson_r_deception_organism": pooled_r,
                }
            }
            print(f"  {'POOLED':10s} n={len(all_dec):>4}  "
                  f"dec_auc={pooled_dec_auc:.4f}  org_auc={pooled_org_auc:.4f}  "
                  f"r={pooled_r:+.3f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    header = f"{'cell':<22}" + "".join(
        f"{opt:>12}" for opt in args.options
        for _ in range(3))
    print(header)
    sub_header = f"{'':22}" + "".join(
        f" {'dec_auc':>4} {'org_auc':>4} {'r':>4}" for _ in args.options)
    print(sub_header)

    for cell in wanted:
        row = f"{cell:<22}"
        for opt in args.options:
            if cell in results and opt in results[cell]:
                p = results[cell][opt]["pooled"]
                row += f" {p['deception_auroc']:>8.4f} {p['organism_auroc']:>8.4f} {p['pearson_r_deception_organism']:>+7.3f}"
            else:
                row += f" {'-':>8} {'-':>8} {'-':>7}"
        print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "device": DEVICE, "seed": SEED, "max_epochs": args.max_epochs,
        "organism_loss_weight": args.organism_loss_weight,
        "options": args.options, "cells": wanted, "results": results,
    }, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
