#!/usr/bin/env python3
"""Compare the v3.1, v3.2 and v3.3 scoring rules on the local judge-probe cache
across all 21 dev datasets.  No NDIF needed — uses the same cached judge margins
and probe activations as ``score_full_eval.py`` and the constant-fitting scripts.

Output: one row per dataset, columns show judge-alone and probe-alone AUROC, the
three rules' AUROC, and the mean per-row agreement signal (v3.2's modulation
signal).

READ THIS BEFORE USING IT TO SELECT A RULE.  Dev cannot discriminate the v3.3
regime split.  Its base-model datasets are still deception-organism data, so the
probe transfers to them fine and the base/LoRA regimes look identical here
(mean agreement 0.285 vs 0.290).  Dev is also saturated: judge-alone AUROC
averages 0.9528 and many units sit above 0.999.  Treat this as a WIRING CHECK —
it catches sign flips, bad standardization and collapsed branches.  It does not
predict the official Notus/Iris outcome, and the probe is in-sample on every dev
row, so probe-alone numbers here are optimistic.
"""

import csv, json, sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# -- load constants -----------------------------------------------------------
v31 = json.load(open(REPO / "results/ensemble_gate_eval/sonic_v3_1_constants.json"))
v32 = json.load(open(REPO / "results/ensemble_gate_eval/sonic_v3_2_constants.json"))

# -- load judge cache ---------------------------------------------------------
judge_cache = json.load(open(REPO / "experiments/ensemble_gate_eval/direct_judge_cache.json"))
margins = {}
for k, (l0, l1) in judge_cache.items():
    ds, idx = k.split("|", 1)
    margins[(ds, idx)] = float(l1) - float(l0)

# -- load probe logits --------------------------------------------------------
import torch
sys.path.insert(0, str(REPO / "experiments/nonlinear_probe"))
from token_probes import TokenProbe, build_token_probe_model

LAYER = 46
PROBE_DIR = {"gemma": "gemma_probe", "qwen": "qwen_probe", "nemotron": "nemotron_probe"}
HIDDEN_DIM = {"gemma": 5376, "qwen": 5120, "nemotron": 4096}

probes = {}
for fam in ("gemma", "qwen", "nemotron"):
    d = REPO / "submission/whitebox_probe" / PROBE_DIR[fam]
    cfg = json.load(open(d / "config.json"))
    net = build_token_probe_model(cfg["architecture"], HIDDEN_DIM[fam])
    net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    net.eval()
    probe = TokenProbe(cfg["architecture"], seed=0, device="cpu", batch_token_budget=8192)
    probe.model = net
    probe.feature_mean = torch.load(d / "feature_mean.pt", map_location="cpu")
    probe.feature_std = torch.load(d / "feature_std.pt", map_location="cpu")
    probes[fam] = probe

# -- scoring rules ------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))

def score_v31(judge_z, probe_z, cap):
    return judge_z + cap * np.tanh(probe_z)

def score_v32(judge_z, probe_z, base_cap, max_cap, agree_scale):
    raw = judge_z * probe_z / max(agree_scale, 1e-8)
    agreement = np.clip(raw, 0.0, 1.0)
    cap = base_cap + agreement * (max_cap - base_cap)
    return judge_z + cap * np.tanh(probe_z)

# -- evaluate -----------------------------------------------------------------
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

manifest = list(csv.DictReader(open(REPO / "dev_splits/manifest.csv")))
by_dataset = {}
for row in manifest:
    ds = row["dataset"]
    idx = str(row["index"])
    if (ds, idx) not in margins:
        continue
    by_dataset.setdefault(ds, []).append(
        (idx, row["deceptive"] == "True", margins[(ds, idx)], bool(row["lora_id"])))

judge_sd = v31["judge_margin_sd"]
# v3.1 constants
cap31 = v31["probe_cap"]
gain = v31["probe_gain"]
mean31 = v31["probe_logit_mean"]
sd31 = v31["probe_logit_sd"]
# v3.2 constants
base_cap = v32["base_cap"]
max_cap = v32["max_cap"]
agree_scale = v32["agreement_scale"]
mean32 = v32["probe_logit_mean"]
sd32 = v32["probe_logit_sd"]

results = []
mean_agreements = []

print(f"{'dataset':<56}{'n':>5}{'judge':>8}{'probe':>8}{'v31':>8}{'v32':>8}"
      f"{'pz_sd':>8}{'rho':>8}{'best':>7}  regime")
print("-" * 120)

for ds in sorted(by_dataset):
    rows = by_dataset[ds]
    regime = "LoRA" if rows[0][3] else "base"
    family = "nemotron" if "Nemotron" in ds else ("gemma" if "gemma" in ds else "qwen")

    indices = [r[0] for r in rows]
    labels = np.array([r[1] for r in rows])
    jm = np.array([r[2] for r in rows])

    cache_fname = REPO / f"results/whitebox/activations/{ds.replace('/', '__')}.tokens.npz"
    data = dict(np.load(cache_fname, allow_pickle=True))
    tokens_key = [k for k in data if k.startswith(f"tokens_L{LAYER}")][0]
    all_tokens = data[tokens_key]
    offsets = data["token_offsets"].astype(np.int64)
    ds_indices = data["index"]

    idx_map = {str(ds_indices[r]): r for r in range(len(ds_indices))}
    present = [i for i in indices if i in idx_map]
    row_ids = np.array([idx_map[i] for i in present])
    labels = np.array([labels[p] for p in range(len(indices)) if indices[p] in idx_map])
    jm = np.array([jm[p] for p in range(len(indices)) if indices[p] in idx_map])

    pieces = [all_tokens[offsets[r]:offsets[r + 1]] for r in row_ids]
    lengths = [len(p) for p in pieces]
    flat_np = np.concatenate(pieces, axis=0)
    new_offs = np.cumsum([0] + lengths).astype(np.int64)
    flat_t = torch.from_numpy(flat_np).float()
    probe_logits = np.nan_to_num(probes[family].decision_function(flat_t, new_offs), nan=0.0)

    # -- v3.1 score (uses v3.1 per-family mean/sd) ----------------------------
    pz31 = (probe_logits - mean31.get(family, v31["default_probe_mean"])) / max(
        sd31.get(family, v31["default_probe_sd"]), 1e-8)
    jz = jm / max(judge_sd, 1e-8)
    s31 = score_v31(jz, pz31, cap31)

    # -- v3.2 score (uses v3.2 per-family mean/sd) ----------------------------
    pz32 = (probe_logits - mean32.get(family, v32["default_probe_mean"])) / max(
        sd32.get(family, v32["default_probe_sd"]), 1e-8)
    s32 = score_v32(jz, pz32, base_cap, max_cap, agree_scale)

    # agreement signal (v3.2 diagnostic)
    raw_ag = np.clip(jz * pz32 / max(agree_scale, 1e-8), 0.0, 1.0)
    mean_ag = float(np.mean(raw_ag))

    # Rank agreement between the two detectors.  Tested as a label-free way to
    # pick which detector should lead; it does NOT separate the two cases (see
    # the module docstring).  Kept as a diagnostic.
    rho = float(spearmanr(jz, pz32).statistic)

    auroc = lambda s: roc_auc_score(labels, s) if len(np.unique(labels)) > 1 else float("nan")

    aj = auroc(jz)
    ap = auroc(pz32)
    a31 = auroc(s31)
    a32 = auroc(s32)

    spread = float(np.std(pz32))
    # Which detector SHOULD have led, with labels -- the target rho must predict.
    leader = "probe" if ap > aj else "judge"

    if not np.isnan(aj):
        results.append((aj, ap, a31, a32, mean_ag, regime, len(labels), rho, leader))

    marker = ""
    if not np.isnan(a32) and not np.isnan(a31):
        diff = a32 - a31
        if abs(diff) > 0.001:
            marker = f"  *** {diff:+.4f}"

    ds_short = ds.split("/")[-1]
    print(f"{ds_short:<56}{len(labels):>5}{aj:>8.4f}{ap:>8.4f}{a31:>8.4f}{a32:>8.4f}"
          f"{spread:>8.2f}{rho:>8.3f}{leader:>7}  {regime}{marker}")
    mean_agreements.append(mean_ag)

# -- summary ------------------------------------------------------------------
js = [r[0] for r in results]
ps = [r[1] for r in results]
s31s = [r[2] for r in results]
s32s = [r[3] for r in results]
ags = [r[4] for r in results]
regimes = [r[5] for r in results]
rhos = [r[7] for r in results]
leaders = [r[8] for r in results]

print(f"\n{'mean':<56}{'':>5}{np.mean(js):>8.4f}{np.mean(ps):>8.4f}{np.mean(s31s):>8.4f}"
      f"{np.mean(s32s):>8.4f}{'':>8}{np.mean(rhos):>8.3f}")

for regime in ("base", "LoRA"):
    idxs = [i for i, r in enumerate(regimes) if r == regime]
    if not idxs:
        continue
    print(f"  {regime:<6} n={len(idxs):>2}{'':>2}"
          f"{np.mean([js[i] for i in idxs]):>8.4f}{np.mean([ps[i] for i in idxs]):>8.4f}"
          f"{np.mean([s31s[i] for i in idxs]):>8.4f}{np.mean([s32s[i] for i in idxs]):>8.4f}"
          f"{'':>8}{np.mean([rhos[i] for i in idxs]):>8.3f}"
          f"  agreement={np.mean([ags[i] for i in idxs]):.3f}")

# Negative result, kept deliberately: rank agreement does NOT tell you which
# detector should lead.  If it did, the rho ranges below would be disjoint.
judge_rho = [rhos[i] for i, lead in enumerate(leaders) if lead == "judge"]
probe_rho = [rhos[i] for i, lead in enumerate(leaders) if lead == "probe"]
if judge_rho and probe_rho:
    print(f"\nrho where the judge should lead: {min(judge_rho):.3f}--{max(judge_rho):.3f} "
          f"(n={len(judge_rho)})")
    print(f"rho where the probe should lead: {min(probe_rho):.3f}--{max(probe_rho):.3f} "
          f"(n={len(probe_rho)})")
    print("ranges overlap -> rho cannot select the leader; see the module docstring.")
