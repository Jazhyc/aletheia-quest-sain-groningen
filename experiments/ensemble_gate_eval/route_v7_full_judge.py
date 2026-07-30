#!/usr/bin/env python3
"""V7 router with full multi-route Phoenix v6.3 judge (not single-route)."""
import csv, json, sys, math
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/nonlinear_probe"))

ACTIVATIONS = REPO / "results/whitebox/activations"
MANIFEST = REPO / "dev_splits/manifest.csv"
HIDDEN_DIM = {"gemma": 5376, "qwen": 5120, "nemotron": 4096}
LAYER = 46
D_MODEL = 128


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


def load_family_probe(family):
    probe_dir = REPO / "submission/whitebox_probe" / f"{family}_probe"
    cfg = json.load(open(probe_dir / "config.json"))
    net = TransformerTokenProbe(
        hidden_dim=HIDDEN_DIM[family], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], dim_feedforward=cfg["dim_feedforward"],
        n_blocks=cfg["n_blocks"], dropout=cfg["dropout"])
    net.load_state_dict(torch.load(probe_dir / "model.pt", map_location="cpu"))
    net.eval()
    feature_mean = torch.load(probe_dir / "feature_mean.pt", map_location="cpu")
    feature_std = torch.load(probe_dir / "feature_std.pt", map_location="cpu")
    return net, feature_mean, feature_std


def load_organism_head(family):
    """Load per-family organism head with scaler folded in."""
    head_path = REPO / "submission/whitebox_probe" / f"{family}_organism_head.pt"
    scaler_path = REPO / "submission/whitebox_probe" / f"{family}_organism_scaler.npz"
    if not head_path.exists():
        return None
    state = torch.load(head_path, map_location="cpu")
    head = torch.nn.Linear(D_MODEL, 1)
    head.load_state_dict(state)
    if scaler_path.exists():
        sd = np.load(scaler_path)
        mean = torch.tensor(sd["mean"], dtype=torch.float32)
        scale = torch.tensor(sd["scale"], dtype=torch.float32)
        with torch.no_grad():
            head.weight.data = head.weight.data / scale.unsqueeze(0)
            head.bias.data = head.bias.data - (head.weight.data * mean.unsqueeze(0)).sum(dim=1)
    head.eval()
    return head


# Load judges
print("Loading judge scores...")
judge_cache = json.load(open(REPO / "experiments/ensemble_gate_eval/v63_full_judge_cache.json"))

# Build judge lookup
judge_map = {}
for ds_name, rows in judge_cache.items():
    for r in rows:
        judge_map[(ds_name, r["index"])] = r["score"]

# Load manifest labels + lora info
manifest_rows = list(csv.DictReader(open(MANIFEST)))
label_map = {}
for row in manifest_rows:
    label_map[(row["dataset"], int(row["index"]))] = (
        row["deceptive"] == "True",
        bool(row.get("lora_id", "")))

# Load probes (cpu for now)
print("Loading probes...")
probes = {}
for fam in ("gemma", "qwen", "nemotron"):
    net, fmean, fstd = load_family_probe(fam)
    probes[fam] = (net, fmean, fstd)

organism_heads = {}
for fam in ("qwen", "gemma"):
    oh = load_organism_head(fam)
    if oh is not None:
        organism_heads[fam] = oh
        print(f"  {fam} organism head loaded")
print(f"  nemotron: no organism head")

# Map datasets to families
ds_families = {}
for row in manifest_rows:
    ds = row["dataset"]
    if ds in ds_families:
        continue
    model_id = row.get("model_id", "")
    for fam in ("gemma", "qwen", "nemotron"):
        if fam in model_id.lower():
            ds_families[ds] = fam
            break

# Process each dataset
device = "cpu"
all_results = []

for ds_name in sorted(ds_families):
    family = ds_families[ds_name]
    short = ds_name.split("/")[-1]
    net, fmean, fstd = probes[family]
    org_head = organism_heads.get(family)

    # Find activation file
    act_fname = ACTIVATIONS / f"{ds_name.replace('/', '__')}.tokens.npz"
    if not act_fname.exists():
        print(f"  {short}: no activations, skip")
        continue

    print(f"  {short} ({family})...", end=" ", flush=True)

    data = dict(np.load(act_fname, allow_pickle=True))
    tokens_key = [k for k in data if k.startswith(f"tokens_L{LAYER}")][0]
    all_tokens = data[tokens_key]
    offsets = data["token_offsets"].astype(np.int64)
    ds_indices = data["index"]

    # Get labels and judge scores aligned with activation indices
    N = len(ds_indices)
    labels = np.zeros(N, dtype=bool)
    judge_scores = np.full(N, np.nan)
    is_org = np.zeros(N, dtype=bool)

    for i in range(N):
        idx = int(ds_indices[i]) if hasattr(ds_indices[i], 'item') else int(ds_indices[i])
        lbl, org = label_map.get((ds_name, idx), (False, False))
        labels[i] = lbl
        is_org[i] = org
        js = judge_map.get((ds_name, idx), None)
        if js is not None:
            judge_scores[i] = js

    # Skip if no judge scores
    if np.all(np.isnan(judge_scores)):
        print("no judge scores, skip")
        continue

    # Score with probe + organism head
    pieces = [all_tokens[offsets[i]:offsets[i + 1]] for i in range(N)]
    lengths = [len(p) for p in pieces]
    flat_np = np.concatenate(pieces, axis=0)
    new_offs = np.cumsum([0] + lengths).astype(np.int64)
    flat_t = torch.from_numpy(flat_np).float()

    with torch.no_grad():
        # Batch by token budget
        batch_size = 32
        probe_scores = np.zeros(N, dtype=np.float64)
        org_confs = np.zeros(N, dtype=np.float64)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            rows = list(range(start, end))
            ml = max(lengths[r] for r in rows)
            h = flat_t.shape[1]
            padded = torch.zeros(len(rows), ml, h, dtype=torch.float32)
            mask = torch.zeros(len(rows), ml, dtype=torch.bool)
            for pos, row in enumerate(rows):
                s, e = int(new_offs[row]), int(new_offs[row + 1])
                padded[pos, :e - s] = flat_t[s:e]
                mask[pos, :e - s] = True
            x = (padded - fmean) / fstd
            x = x * mask.unsqueeze(-1)
            logits, pooled = net(x, mask, return_pooled=True)
            for pos, row in enumerate(rows):
                probe_scores[row] = float(torch.sigmoid(logits[pos]).item())
            if org_head is not None:
                org_raw = org_head(pooled).squeeze(-1)
                for pos, row in enumerate(rows):
                    org_confs[row] = float(torch.sigmoid(org_raw[pos]).item())

    # Apply router at t=0.5 and t=0.7
    for t_val, t_label in [(0.5, "t=0.5"), (0.7, "t=0.7")]:
        use_probe = org_confs > t_val
        router_scores = np.where(use_probe, probe_scores, judge_scores)

        # AUROC (overall)
        valid = ~np.isnan(judge_scores)
        if len(np.unique(labels[valid])) > 1:
            a_all = roc_auc_score(labels[valid], router_scores[valid])
        else:
            a_all = float("nan")

        # Split base vs organisms
        base_v = valid & ~is_org
        org_v = valid & is_org
        a_base = roc_auc_score(labels[base_v], router_scores[base_v]) if len(np.unique(labels[base_v])) > 1 else float("nan")
        a_org = roc_auc_score(labels[org_v], router_scores[org_v]) if len(np.unique(labels[org_v])) > 1 else float("nan")

        n_probe = int(use_probe[valid].sum())
        n_judge = valid.sum() - n_probe

        all_results.append({
            "dataset": short, "family": family, "threshold": t_label,
            "n": valid.sum(), "auroc_all": a_all, "auroc_base": a_base,
            "auroc_org": a_org, "n_probe": n_probe, "n_judge": n_judge,
            "n_base": base_v.sum(), "n_org": org_v.sum(),
        })

    print(f"done")

# Aggregate
print("\n=== Results ===")
for t_label in ["t=0.5", "t=0.7"]:
    tres = [r for r in all_results if r["threshold"] == t_label]
    a_all = [r["auroc_all"] for r in tres if not np.isnan(r["auroc_all"])]
    a_base = [r["auroc_base"] for r in tres if not np.isnan(r["auroc_base"])]
    a_org = [r["auroc_org"] for r in tres if not np.isnan(r["auroc_org"])]

    # Pooled AUROC
    all_l, all_s, all_org = [], [], []
    for r in tres:
        ds = next(d for d in ds_families if d.split("/")[-1] == r["dataset"])
        # Recompute pooled from raw data... simpler: aggregate the mean
        pass

    print(f"\n{t_label}:")
    print(f"  All datasets:     AUROC={np.mean(a_all):.4f} (n={len(a_all)})")
    print(f"  Base models only: AUROC={np.mean(a_base):.4f} (n={len(a_base)})")
    print(f"  Organisms only:   AUROC={np.mean(a_org):.4f} (n={len(a_org)})")
    total_probe = sum(r["n_probe"] for r in tres)
    total_judge = sum(r["n_judge"] for r in tres)
    print(f"  Routing: {total_probe} probe, {total_judge} judge")

# Save
out_path = REPO / "results/ensemble_gate_eval/v7_router_full_judge.json"
json.dump(all_results, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")

# Compute pooled AUROC properly
print("\n=== Pooled AUROC ===")
for t_label in ["t=0.5", "t=0.7"]:
    all_labels, all_scores, all_is_org = [], [], []
    for ds_name in sorted(ds_families):
        family = ds_families[ds_name]
        short = ds_name.split("/")[-1]
        act_fname = ACTIVATIONS / f"{ds_name.replace('/', '__')}.tokens.npz"
        if not act_fname.exists():
            continue
        data = dict(np.load(act_fname, allow_pickle=True))
        tokens_key = [k for k in data if k.startswith(f"tokens_L{LAYER}")][0]
        all_tokens = data[tokens_key]
        offsets = data["token_offsets"].astype(np.int64)
        ds_indices = data["index"]

        net, fmean, fstd = probes[family]
        org_head = organism_heads.get(family)
        t_val = 0.5 if t_label == "t=0.5" else 0.7

        N = len(ds_indices)
        pieces = [all_tokens[offsets[i]:offsets[i + 1]] for i in range(N)]
        lengths = [len(p) for p in pieces]
        flat_np = np.concatenate(pieces, axis=0)
        new_offs = np.cumsum([0] + lengths).astype(np.int64)
        flat_t = torch.from_numpy(flat_np).float()

        with torch.no_grad():
            batch_size = 32
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                rows = list(range(start, end))
                ml = max(lengths[r] for r in rows)
                h = flat_t.shape[1]
                padded = torch.zeros(len(rows), ml, h, dtype=torch.float32)
                mask = torch.zeros(len(rows), ml, dtype=torch.bool)
                for pos, row in enumerate(rows):
                    s, e = int(new_offs[row]), int(new_offs[row + 1])
                    padded[pos, :e - s] = flat_t[s:e]
                    mask[pos, :e - s] = True
                x = (padded - fmean) / fstd
                x = x * mask.unsqueeze(-1)
                logits, pooled = net(x, mask, return_pooled=True)
                for pos, row in enumerate(rows):
                    idx = int(ds_indices[row]) if hasattr(ds_indices[row], 'item') else int(ds_indices[row])
                    lbl, org = label_map.get((ds_name, idx), (False, False))
                    js = judge_map.get((ds_name, idx))
                    if js is None:
                        continue
                    ps = float(torch.sigmoid(logits[pos]).item())
                    if org_head is not None:
                        oc = float(torch.sigmoid(org_head(pooled[pos]).item()))
                    else:
                        oc = 0.0
                    rs = ps if oc > t_val else js
                    all_labels.append(lbl)
                    all_scores.append(rs)
                    all_is_org.append(org)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_is_org = np.array(all_is_org)

    a_all = roc_auc_score(all_labels, all_scores)
    base_m = ~all_is_org
    org_m = all_is_org
    a_base = roc_auc_score(all_labels[base_m], all_scores[base_m])
    a_org = roc_auc_score(all_labels[org_m], all_scores[org_m])

    print(f"\n{t_label}:")
    print(f"  All datasets:     AUROC={a_all:.4f} (n={len(all_labels)})")
    print(f"  Base models only: AUROC={a_base:.4f} (n={base_m.sum()})")
    print(f"  Organisms only:   AUROC={a_org:.4f} (n={org_m.sum()})")
