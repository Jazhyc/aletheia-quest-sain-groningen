#!/usr/bin/env python3
"""Compare SAE-denoised probe vs raw probe on leave-one-cell-out folds.

Qwen only (official SAEs are for Qwen3.5-27B). Tests three signals:
  A) Raw-activation baseline (standard TransformerTokenProbe on L46).
  B) SAE-denoised: reconstruct activations from top-50 SAE features, then
     train the same probe architecture on the reconstructed activations.
  C) SAE feature classifier: mean SAE feature activation per row (sparse
     80K-d), logistic regression.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments/ensemble_gate_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments/nonlinear_probe"))
sys.path.insert(0, str(REPO_ROOT / "submission"))

from token_probes import TokenProbe
from run_eval import BaseModelData, load_manifest_rows
from train_v3_3_probe import cell_of, subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# SAE (kept in float32 on CPU; encoding done in float32)
# ---------------------------------------------------------------------------

def load_sae(layer: int = 46) -> dict:
    from huggingface_hub import hf_hub_download
    repo = f"Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"
    path = hf_hub_download(repo, f"layer{layer}.sae.pt")
    return torch.load(path, map_location="cpu")  # all float32


@torch.no_grad()
def sae_encode_batch(x: torch.Tensor, sae: dict, k: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (n_tokens, 5120) float32, CPU. Returns (values, indices) float32/int64."""
    b_dec = sae["b_dec"]           # (5120,)
    W_enc = sae["W_enc"]           # (81920, 5120)
    b_enc = sae["b_enc"]           # (81920,)
    x_centered = x - b_dec.unsqueeze(0)
    pre_acts = x_centered @ W_enc.T + b_enc.unsqueeze(0)
    return torch.topk(pre_acts, k, dim=-1)


@torch.no_grad()
def sae_decode_batch(indices: torch.Tensor, values: torch.Tensor, sae: dict) -> torch.Tensor:
    """indices/values: (n_tokens, k). Returns (n_tokens, 5120)."""
    W_dec = sae["W_dec"]           # (5120, 81920)
    b_dec = sae["b_dec"]           # (5120,)
    d_sae = W_dec.shape[1]
    n = indices.shape[0]
    sparse = torch.zeros(n, d_sae)
    sparse.scatter_(1, indices, values)
    return sparse @ W_dec.T + b_dec.unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results/shadow/sae_probe_compare.json")
    args = parser.parse_args(argv)

    manifest = load_manifest_rows()
    qwen = BaseModelData("qwen", manifest)
    cells = cell_of(qwen.organisms)
    cell_names = sorted(set(cells.tolist()))

    print("Loading SAE (layer 46)...", flush=True)
    sae = load_sae(46)
    print(f"SAE loaded: d_sae={sae['W_enc'].shape[0]}", flush=True)

    # Map cell name to readable label
    cell_label = {
        "instr/adapter": "instr/adapter",
        "instr/base": "instr/base",
        "varied/adapter": "varied/adapter",
        "varied/base": "varied/base",
    }

    results = {}
    for held_out_cell in cell_names:
        if held_out_cell == "instr/base":
            continue  # nemotron only, no qwen

        train_rows = np.flatnonzero(cells != held_out_cell)
        test_rows = np.flatnonzero(cells == held_out_cell)
        labels = qwen.labels[test_rows]

        print(f"\n{'='*60}")
        print(f"cell {held_out_cell}: train={len(train_rows)} test={len(test_rows)}")

        # --- Common: pre-compute SAE encodings for all rows ---
        flat_all, offs_all = subset(qwen, np.arange(len(qwen.labels)))
        print(f"Encoding {len(flat_all)} tokens through SAE...", flush=True)
        all_vals, all_idx = [], []
        for bi in range(0, len(flat_all), 2048):
            batch = flat_all[bi:bi+2048].cpu()  # float32 CPU
            idx, val = sae_encode_batch(batch, sae, k=50)
            all_idx.append(idx)
            all_vals.append(val)
            if bi % 8192 == 0:
                print(f"  {bi}/{len(flat_all)} tokens", flush=True)
        all_idx = torch.cat(all_idx)
        all_vals = torch.cat(all_vals)
        print(f"  done: {len(all_idx)} token encodings", flush=True)

        # --- A) Raw probe ---
        print("Training raw probe...", flush=True)
        flat_train, offs_train = subset(qwen, train_rows)
        tp = TokenProbe("transformer", seed=args.seed, device=DEVICE,
                        max_epochs=args.max_epochs)
        tp.fit(flat_train, offs_train, qwen.labels[train_rows])
        flat_test, offs_test = subset(qwen, test_rows)
        logits_raw = tp.predict_proba(flat_test, offs_test)[:, 1]
        eps = 1e-12
        logits_raw = np.log(np.clip(logits_raw, eps, 1-eps) / np.clip(1-logits_raw, eps, 1-eps))
        auroc_raw = float(roc_auc_score(labels, logits_raw))
        print(f"  raw AUROC: {auroc_raw:.4f}", flush=True)

        # --- B) SAE-denoised probe ---
        print("Reconstructing activations from SAE...", flush=True)
        rec_tokens = torch.zeros_like(flat_all)  # float32 CPU
        for bi in range(0, len(all_idx), 256):
            ei = min(bi+256, len(all_idx))
            rec_tokens[bi:ei] = sae_decode_batch(all_idx[bi:ei], all_vals[bi:ei], sae)
        rec_tokens = rec_tokens.to(DEVICE)

        # Build per-row tensors for train/test
        def build_flat(tokens, offsets, row_indices):
            lengths = [int(offsets[i+1] - offsets[i]) for i in row_indices]
            out = torch.empty((sum(lengths), tokens.shape[1]), dtype=tokens.dtype, device=DEVICE)
            new_offs = np.zeros(len(row_indices)+1, dtype=np.int64)
            pos = 0
            for j, ri in enumerate(row_indices):
                start, end = int(offsets[ri]), int(offsets[ri+1])
                n = end - start
                out[pos:pos+n] = tokens[start:end]
                new_offs[j+1] = new_offs[j] + n
                pos += n
            return out, new_offs

        flat_train_rec, offs_train_rec = build_flat(rec_tokens, offs_all, train_rows)
        flat_test_rec, offs_test_rec = build_flat(rec_tokens, offs_all, test_rows)

        print("Training SAE-denoised probe...", flush=True)
        tp_rec = TokenProbe("transformer", seed=args.seed, device=DEVICE,
                            max_epochs=args.max_epochs)
        tp_rec.fit(flat_train_rec, offs_train_rec, qwen.labels[train_rows])
        logits_rec = tp_rec.predict_proba(flat_test_rec, offs_test_rec)[:, 1]
        logits_rec = np.log(np.clip(logits_rec, eps, 1-eps) / np.clip(1-logits_rec, eps, 1-eps))
        auroc_rec = float(roc_auc_score(labels, logits_rec))
        print(f"  SAE-denoised AUROC: {auroc_rec:.4f}  (delta: {auroc_rec - auroc_raw:+.4f})", flush=True)

        # --- C) SAE feature classifier (logistic regression on mean features) ---
        print("Computing SAE feature means...", flush=True)
        d_sae = sae["W_enc"].shape[0]
        feature_means_np = np.zeros((len(qwen.labels), d_sae), dtype=np.float32)
        row_offset = 0
        for i in range(len(qwen.labels)):
            start, end = int(offs_all[i]), int(offs_all[i+1])
            n = end - start
            if n > 0:
                ri = all_idx[row_offset:row_offset+n]
                rv = all_vals[row_offset:row_offset+n]
                # Mean over tokens: sparse average
                for t in range(n):
                    feature_means_np[i, ri[t].long()] += rv[t].item()
                feature_means_np[i] /= n
            row_offset += n

        # Filter to features active in training
        active = feature_means_np[train_rows].sum(axis=0) > 0
        n_active = int(active.sum())
        print(f"  active features: {n_active} / {d_sae}", flush=True)

        X_train = feature_means_np[train_rows][:, active]
        X_test = feature_means_np[test_rows][:, active]
        y_train = qwen.labels[train_rows]

        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=args.seed, solver='saga')
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        clf.class_weight = {0: len(y_train)/(2*max(n_neg,1)), 1: len(y_train)/(2*max(n_pos,1))}
        clf.fit(X_train, y_train)
        logits_feat = clf.decision_function(X_test)
        auroc_feat = float(roc_auc_score(labels, logits_feat))
        print(f"  SAE features AUROC: {auroc_feat:.4f}  (delta: {auroc_feat - auroc_raw:+.4f})", flush=True)

        results[held_out_cell] = {
            "n_train": int(len(train_rows)),
            "n_test": int(len(test_rows)),
            "raw": auroc_raw,
            "sae_denoised": auroc_rec,
            "sae_features": auroc_feat,
        }

        # Free memory
        del flat_all, offs_all, all_idx, all_vals, rec_tokens
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"{'cell':<16s}  {'raw':>8s}  {'SAE-denoised':>13s}  {'SAE-features':>13s}  {'delta-denoised':>14s}  {'delta-features':>14s}")
    for cell, r in results.items():
        dd = r["sae_denoised"] - r["raw"]
        df = r["sae_features"] - r["raw"]
        print(f"{cell:<16s}  {r['raw']:8.4f}  {r['sae_denoised']:13.4f}  {r['sae_features']:13.4f}  {dd:+14.4f}  {df:+14.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
