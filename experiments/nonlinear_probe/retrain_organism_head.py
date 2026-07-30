#!/usr/bin/env python3
"""Retrain organism head on CURRENT encoder's pooled features.

Compares:
  A) Per-family LogisticRegression heads (qwen, gemma separately)
  B) Shared LogisticRegression head (qwen+gemma combined)

Train/test split: leave-one-organism-out per family to test generalization
to unseen adapter types. Also tests balanced vs imbalanced training.

Outputs:
  - Per-family AUROC on held-out organisms
  - Shared head AUROC on held-out organisms
  - Best head saved to submission/whitebox_probe/shared_organism_head.pt
"""

import json, math, sys, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/nonlinear_probe"))

ACTIVATIONS = REPO / "results/whitebox/activations"
MANIFEST = REPO / "dev_splits/manifest.csv"
PROBE_DIR = {"gemma": "gemma_probe", "qwen": "qwen_probe"}
HIDDEN_DIM = {"gemma": 5376, "qwen": 5120}
LAYER = 46
BATCH_TOKEN_BUDGET = 8192
DEVICE = "cpu"


def sinusoidal_position_encoding(seq_len, d_model, device=None):
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                         * (-math.log(10000.0) / d_model))
    enc = torch.zeros(seq_len, d_model, device=device)
    enc[:, 0::2] = torch.sin(position * div_term)
    cc = enc[:, 1::2].shape[1]
    enc[:, 1::2] = torch.cos(position * div_term)[:, :cc]
    return enc


class TransformerTokenProbe(torch.nn.Module):
    def __init__(self, hidden_dim, d_model=128, n_heads=4, dim_feedforward=256, n_blocks=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.projection = torch.nn.Linear(hidden_dim, d_model)
        block = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(block, num_layers=n_blocks)
        self.head = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(d_model, 1))

    def forward(self, padded_tokens, padding_mask, return_pooled=False):
        seq_len = padded_tokens.shape[1]
        pe = sinusoidal_position_encoding(seq_len, self.d_model, device=padded_tokens.device)
        x = self.projection(padded_tokens) + pe.unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=~padding_mask)
        m = padding_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        logit = self.head(pooled).squeeze(-1)
        if return_pooled:
            return logit, pooled
        return logit


def load_probe(family):
    d = REPO / "submission/whitebox_probe" / PROBE_DIR[family]
    cfg = json.loads((d / "config.json").read_text())
    net = TransformerTokenProbe(
        hidden_dim=HIDDEN_DIM[family], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], dim_feedforward=cfg["dim_feedforward"],
        n_blocks=cfg["n_blocks"], dropout=cfg["dropout"])
    net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    net.eval()
    return net, torch.load(d / "feature_mean.pt", map_location="cpu"), torch.load(d / "feature_std.pt", map_location="cpu")


def extract_pooled(flat_features, offsets, probe, fmean, fstd):
    """Extract pooled(128) features for all rows."""
    N = len(offsets) - 1
    lengths = (offsets[1:] - offsets[:-1]).tolist()
    order = sorted(range(N), key=lambda p: lengths[p])
    batches, current = [], []
    for pos in order:
        w = lengths[pos]
        if current and (len(current)+1)*max(lengths[p] for p in current+[pos]) > BATCH_TOKEN_BUDGET:
            batches.append(current); current = []
        current.append(pos)
    if current: batches.append(current)

    all_pooled = np.zeros((N, 128), dtype=np.float32)
    with torch.no_grad():
        for row_ids in batches:
            ml = max(lengths[r] for r in row_ids)
            h = flat_features.shape[1]
            padded = torch.zeros(len(row_ids), ml, h, dtype=torch.float32, device=DEVICE)
            mask = torch.zeros(len(row_ids), ml, dtype=torch.bool, device=DEVICE)
            for pos, row in enumerate(row_ids):
                s, e = int(offsets[row]), int(offsets[row+1])
                padded[pos, :e-s] = flat_features[s:e]
                mask[pos, :e-s] = True
            x = (padded.to(DEVICE) - fmean.to(DEVICE)) / fstd.to(DEVICE)
            x = x * mask.unsqueeze(-1)
            _, pooled = probe(x, mask, return_pooled=True)
            pooled_np = pooled.cpu().numpy()
            # Handle NaN from gemma normalization
            pooled_np = np.nan_to_num(pooled_np, nan=0.0, posinf=0.0, neginf=0.0)
            for pos, row in enumerate(row_ids):
                all_pooled[row] = pooled_np[pos]
    return all_pooled


def main():
    print("=" * 80)
    print("Organism head retraining — shared vs per-family comparison")
    print("=" * 80)

    # Load manifest
    manifest = list(csv.DictReader(open(MANIFEST)))
    by_dataset = defaultdict(list)
    for row in manifest:
        by_dataset[row["dataset"]].append(row)

    # Load probes
    probes = {}
    for fam in ["qwen", "gemma"]:
        probes[fam] = load_probe(fam)
    print("Probes loaded: qwen, gemma")

    # --- Extract pooled features for all qwen+gemma rows ---
    all_data = {}  # family -> {dataset_name -> (pooled, is_organism, organism_id)}

    for fam in ["qwen", "gemma"]:
        print(f"\n--- {fam.upper()} ---")
        net, fmean, fstd = probes[fam]
        fam_data = {}

        # Find all datasets for this family
        fam_datasets = [ds for ds in by_dataset if (
            ("gemma" in ds.lower() and fam == "gemma") or
            ("qwen" in ds.lower() and fam == "qwen")
        )]

        for ds_name in sorted(fam_datasets):
            rows = by_dataset[ds_name]
            is_org = any(bool(r["lora_id"]) for r in rows)
            organism_id = rows[0]["lora_id"] or "base"

            cache_fname = ACTIVATIONS / f"{ds_name.replace('/', '__')}.tokens.npz"
            if not cache_fname.exists():
                print(f"  SKIP {ds_name.split('/')[-1][:50]}: no activations")
                continue

            data = dict(np.load(cache_fname, allow_pickle=True))
            tokens_key = [k for k in data if k.startswith(f"tokens_L{LAYER}")][0]
            all_tokens = data[tokens_key]
            offsets = data["token_offsets"].astype(np.int64)
            ds_indices = data["index"]

            idx_map = {str(ds_indices[r]): r for r in range(len(ds_indices))}
            present_rows = [r for r in rows if str(r["index"]) in idx_map]
            if len(present_rows) == 0:
                continue

            row_ids = np.array([idx_map[str(r["index"])] for r in present_rows])
            is_org_arr = np.array([bool(r["lora_id"]) for r in present_rows])

            pieces = [all_tokens[offsets[r]:offsets[r+1]] for r in row_ids]
            lengths = [len(p) for p in pieces]
            flat_np = np.concatenate(pieces, axis=0)
            new_offs = np.cumsum([0] + lengths).astype(np.int64)
            flat_t = torch.from_numpy(flat_np)

            pooled = extract_pooled(flat_t, new_offs, net, fmean, fstd)
            fam_data[ds_name] = (pooled, is_org_arr, organism_id)

            n_org = int(is_org_arr.sum())
            n_base = len(is_org_arr) - n_org
            print(f"  {ds_name.split('/')[-1][:50]:50s} n={len(present_rows):4d}  "
                  f"base={n_base:4d} org={n_org:4d}  pooled_norm={np.linalg.norm(pooled, axis=1).mean():.2f}")

        all_data[fam] = fam_data

    # --- Build train/test splits ---
    # For each family, leave out one organism type as test
    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION")
    print("=" * 80)

    results = {}

    for fam in ["qwen", "gemma"]:
        fam_data = all_data[fam]

        # Get unique organism IDs
        org_ids = sorted(set(oid for _, (_, _, oid) in fam_data.items()))

        # Collect all data
        X_all = []
        y_all = []
        org_id_all = []
        dataset_all = []
        for ds_name, (pooled, is_org, org_id) in fam_data.items():
            X_all.append(pooled)
            y_all.append(is_org.astype(int))
            org_id_all.extend([org_id] * len(is_org))
            dataset_all.extend([ds_name] * len(is_org))

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        org_id_all = np.array(org_id_all)
        n_total = len(y_all)
        n_base = int((y_all == 0).sum())
        n_org = int((y_all == 1).sum())
        print(f"\n{fam.upper()}: {n_total} rows ({n_base} base, {n_org} org), "
              f"{len(set(org_id_all))} organism types")

        # Leave-one-organism-out CV
        unique_orgs = sorted(set(org_id_all))
        per_family_results = []
        shared_results = []

        for held_org in unique_orgs:
            test_mask = org_id_all == held_org
            train_mask = ~test_mask

            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]

            train_base = int((y_train == 0).sum())
            train_org = int((y_train == 1).sum())
            test_base = int((y_test == 0).sum())
            test_org = int((y_test == 1).sum())

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Per-family head
            clf = LogisticRegression(max_iter=10000, class_weight='balanced', C=1.0)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict_proba(X_test_scaled)[:, 1]
            auroc = roc_auc_score(y_test, y_pred)
            ba = balanced_accuracy_score(y_test, y_pred >= 0.5)
            per_family_results.append({
                "held_org": held_org, "auroc": auroc, "ba": ba,
                "test_n": len(y_test), "test_base": test_base, "test_org": test_org,
                "bias": float(clf.intercept_[0]),
            })

            print(f"  {fam} holdout {held_org[:30]:30s}  "
                  f"train={len(y_train)} ({train_base}b/{train_org}o)  "
                  f"test={len(y_test)} ({test_base}b/{test_org}o)  "
                  f"AUROC={auroc:.4f}  BA={ba:.4f}  bias={clf.intercept_[0]:+.3f}")

        if per_family_results:
            mean_auroc = np.mean([r["auroc"] for r in per_family_results])
            mean_ba = np.mean([r["ba"] for r in per_family_results])
            print(f"  {fam} MEAN: AUROC={mean_auroc:.4f}  BA={mean_ba:.4f}  "
                  f"({len(per_family_results)} folds)")

        results[fam] = {
            "per_family_auroc": [r["auroc"] for r in per_family_results],
            "mean_auroc": mean_auroc if per_family_results else float("nan"),
        }

    # --- Shared head (qwen+gemma combined with StandardScaler per family) ---
    print("\n--- SHARED HEAD (qwen+gemma combined) ---")

    # Collect all data with per-family standardization
    X_shared_train = []
    y_shared_train = []
    X_shared_test = []
    y_shared_test = []
    test_info = []

    for fam in ["qwen", "gemma"]:
        fam_data = all_data[fam]
        org_ids_fam = sorted(set(oid for _, (_, _, oid) in fam_data.items()))

        Xf, yf, oif = [], [], []
        for ds_name, (pooled, is_org, org_id) in fam_data.items():
            Xf.append(pooled)
            yf.append(is_org.astype(int))
            oif.extend([org_id] * len(is_org))
        Xf = np.concatenate(Xf, axis=0)
        yf = np.concatenate(yf, axis=0)
        oif = np.array(oif)

        for held_org in org_ids_fam:
            test_mask = oif == held_org
            train_mask = ~test_mask

            X_tr, y_tr = Xf[train_mask], yf[train_mask]
            X_te, y_te = Xf[test_mask], yf[test_mask]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            # Per-family StandardScaler
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            X_shared_train.append(X_tr_s)
            y_shared_train.append(y_tr)
            X_shared_test.append(X_te_s)
            y_shared_test.append(y_te)
            test_info.append((fam, held_org, len(y_te)))

    X_st = np.concatenate(X_shared_train, axis=0)
    y_st = np.concatenate(y_shared_train, axis=0)

    # Train shared head
    shared_clf = LogisticRegression(max_iter=10000, class_weight='balanced', C=1.0)
    shared_clf.fit(X_st, y_st)
    print(f"  Shared head trained on {len(y_st)} rows "
          f"({int((y_st==0).sum())} base, {int((y_st==1).sum())} org)")

    # Evaluate per test fold
    shared_fold_results = []
    for (fam, held_org, n_test), X_te, y_te in zip(test_info, X_shared_test, y_shared_test):
        y_pred = shared_clf.predict_proba(X_te)[:, 1]
        auroc = roc_auc_score(y_te, y_pred)
        ba = balanced_accuracy_score(y_te, y_pred >= 0.5)
        shared_fold_results.append(auroc)
        print(f"  shared holdout {fam}/{held_org[:25]:25s}  n={n_test}  AUROC={auroc:.4f}  BA={ba:.4f}")

    if shared_fold_results:
        mean_shared = np.mean(shared_fold_results)
        print(f"  SHARED MEAN: AUROC={mean_shared:.4f}  ({len(shared_fold_results)} folds)")
        print(f"  Shared head bias: {shared_clf.intercept_[0]:+.4f}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    per_fam_aurocs = []
    for fam in ["qwen", "gemma"]:
        if fam in results:
            a = results[fam]["mean_auroc"]
            print(f"  {fam} per-family LOO:   mean AUROC = {a:.4f}")
            per_fam_aurocs.extend(results[fam]["per_family_auroc"])

    if per_fam_aurocs:
        print(f"  Per-family combined:   mean AUROC = {np.mean(per_fam_aurocs):.4f}")

    if shared_fold_results:
        print(f"  Shared head combined:  mean AUROC = {mean_shared:.4f}")

    # --- Decision ---
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    # Train final heads on ALL data
    print("\nTraining final heads on all data (no holdout)...")

    for approach in ["per_family", "shared"]:
        if approach == "per_family":
            print("\n  Per-family heads:")
            for fam in ["qwen", "gemma"]:
                fam_data = all_data[fam]
                Xf, yf = [], []
                for ds_name, (pooled, is_org, org_id) in fam_data.items():
                    Xf.append(pooled)
                    yf.append(is_org.astype(int))
                Xf = np.concatenate(Xf, axis=0)
                yf = np.concatenate(yf, axis=0)

                scaler = StandardScaler()
                Xf_s = scaler.fit_transform(Xf)
                clf = LogisticRegression(max_iter=10000, class_weight='balanced', C=1.0)
                clf.fit(Xf_s, yf)

                # Save per-family head (weight, bias) + scaler params
                weight = torch.tensor(clf.coef_, dtype=torch.float32)  # (1, 128)
                bias = torch.tensor(clf.intercept_, dtype=torch.float32)  # (1,)

                out_path = REPO / f"submission/whitebox_probe/{fam}_organism_head.pt"
                torch.save({"weight": weight, "bias": bias}, out_path)
                # Also save scaler
                scaler_path = REPO / f"submission/whitebox_probe/{fam}_organism_scaler.npz"
                np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)

                y_pred_full = clf.predict_proba(Xf_s)[:, 1]
                full_auroc = roc_auc_score(yf, y_pred_full)
                print(f"    {fam}: {len(yf)} rows, full-data AUROC={full_auroc:.4f}, "
                      f"bias={clf.intercept_[0]:+.4f}, saved to {out_path.name}")

        else:  # shared
            print("\n  Shared head:")
            X_all_shared, y_all_shared, scalers = [], [], {}
            for fam in ["qwen", "gemma"]:
                fam_data = all_data[fam]
                Xf, yf = [], []
                for ds_name, (pooled, is_org, org_id) in fam_data.items():
                    Xf.append(pooled)
                    yf.append(is_org.astype(int))
                Xf = np.concatenate(Xf, axis=0)
                yf = np.concatenate(yf, axis=0)

                scaler = StandardScaler()
                Xf_s = scaler.fit_transform(Xf)
                X_all_shared.append(Xf_s)
                y_all_shared.append(yf)
                scalers[fam] = scaler

            X_as = np.concatenate(X_all_shared, axis=0)
            y_as = np.concatenate(y_all_shared, axis=0)

            shared_clf_final = LogisticRegression(max_iter=10000, class_weight='balanced', C=1.0)
            shared_clf_final.fit(X_as, y_as)
            weight = torch.tensor(shared_clf_final.coef_, dtype=torch.float32)
            bias = torch.tensor(shared_clf_final.intercept_, dtype=torch.float32)

            out_path = REPO / "submission/whitebox_probe/shared_organism_head_retrained.pt"
            torch.save({"weight": weight, "bias": bias}, out_path)

            # Save per-family scalers
            for fam, scaler in scalers.items():
                scaler_path = REPO / f"submission/whitebox_probe/{fam}_organism_scaler.npz"
                np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)

            y_pred_full = shared_clf_final.predict_proba(X_as)[:, 1]
            full_auroc = roc_auc_score(y_as, y_pred_full)
            print(f"    combined: {len(y_as)} rows, full-data AUROC={full_auroc:.4f}, "
                  f"bias={shared_clf_final.intercept_[0]:+.4f}, saved to {out_path.name}")
            print(f"    Per-family scalers saved to *_organism_scaler.npz")

    print("\nDone.")


if __name__ == "__main__":
    main()
