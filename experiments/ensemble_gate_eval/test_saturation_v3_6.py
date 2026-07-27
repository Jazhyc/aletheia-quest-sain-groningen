"""Why does raising MAX_CAP make Iris worse? Test the tanh saturation hypothesis.

The shipped fusion is::

    score = sigmoid(judge_z + cap * tanh(PROBE_GAIN * probe_z))

with ``PROBE_GAIN = 1.0`` and ``probe_z`` standardised to unit variance. For
``|probe_z| > 2`` the tanh is inside 4% of its asymptote, so every confident
probe row contributes the *same* offset regardless of how confident it is. On a
fold where the probe is strong -- Iris's condition -- most rows sit in that
saturated band, so the probe's magnitude ordering is discarded and only its sign
survives. Raising ``MAX_CAP`` then amplifies a near-binary signal, which is a
plausible reason v3.3 (MAX_CAP 6 steps) scored Iris `0.9393` against v3.5's
`0.9444` at 4 steps on the same probe.

This script scores fusion variants on identical (judge, probe) inputs across
leave-one-organism-out folds, at three probe intensities:

  natural   -- the probe as fitted
  iris_like -- probe sharpened so it leads the judge by ~0.068 AUROC (real Iris)
  notus     -- probe blunted to AUROC 0.56 (measured real Notus quality)

    python experiments/ensemble_gate_eval/test_saturation_v3_6.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ACTIVATIONS = REPO_ROOT / "results/whitebox/activations"
JUDGE_CACHE = HERE / "judge_cache.json"
OUT_JSON = REPO_ROOT / "results/ensemble_gate_eval/saturation_v3_6.json"

SEED = 0
STEP = 0.10418792638565248          # one judge step, = BASE_CAP / 2
BASE_CAP = 2 * STEP
JUDGE_MARGIN_SD = 1.199755138011975


def base_model_of(name: str) -> str:
    if "gemma-3-27b-it" in name:
        return "gemma"
    if "Qwen3.5-27B" in name:
        return "qwen"
    if "Nemotron" in name:
        return "nemotron"
    return "unknown"


def organism_of(name: str) -> str:
    for tag in ("gemma-3-27b-it-", "Qwen3.5-27B-", "A12B-BF16-"):
        if tag in name:
            return name.split(tag, 1)[1]
    return "?"


def load_datasets(judge_cache: dict) -> list[dict]:
    out = []
    for path in sorted(ACTIVATIONS.glob("*.npz")):
        stem = path.name[: -len(".npz")]
        if ".tokens" in stem or ".limit" in stem:
            continue
        blob = np.load(path, allow_pickle=True)
        if "mean_L46" not in blob.files:
            continue
        ds_key = stem.replace("__", "/")
        index = blob["index"]
        judge = np.array(
            [judge_cache.get((ds_key, str(int(i))), (np.nan, np.nan))[1] for i in index],
            dtype=np.float64,
        )
        if not np.isfinite(judge).all():
            continue
        out.append(dict(name=ds_key, base=base_model_of(stem), organism=organism_of(stem),
                        x=blob["mean_L46"].astype(np.float32),
                        y=blob["deceptive"].astype(int), judge=judge))
    return out


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-9)


def retune_auroc(scores, labels, target, rng, sharpen_ok=True):
    """Add noise (or de-noise toward the label) until AUROC hits `target`."""
    z = zscore(scores)
    cur = roc_auc_score(labels, z)
    if target < cur:
        lo, hi = 0.0, 300.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if roc_auc_score(labels, z + mid * rng.standard_normal(z.shape)) > target:
                lo = mid
            else:
                hi = mid
        return zscore(z + ((lo + hi) / 2) * rng.standard_normal(z.shape))
    if not sharpen_ok:
        return z
    signal = np.where(np.asarray(labels) > 0, 1.0, -1.0)
    lo, hi = 0.0, 30.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if roc_auc_score(labels, z + mid * signal) < target:
            lo = mid
        else:
            hi = mid
    return zscore(z + ((lo + hi) / 2) * signal)


# ---- fusion variants -------------------------------------------------------

def fuse(judge_z, probe_z, max_cap, squash, gain):
    agreement = (judge_z * probe_z > 0).astype(np.float64)
    cap = BASE_CAP + agreement * (max_cap - BASE_CAP)
    if squash == "tanh":
        contrib = np.tanh(gain * probe_z)
    elif squash == "clip":
        contrib = np.clip(gain * probe_z, -1.0, 1.0)
    elif squash == "linear":
        contrib = gain * probe_z
    else:
        raise ValueError(squash)
    return judge_z + cap * contrib


VARIANTS = [
    ("v3.5 shipped  tanh g=1.0 cap4", 4 * STEP * 2 / 2, "tanh", 1.0),
    ("tanh g=1.0 cap6",               6 * STEP,          "tanh", 1.0),
    ("tanh g=1.0 cap8",               8 * STEP,          "tanh", 1.0),
    ("tanh g=0.4 cap4",               4 * STEP,          "tanh", 0.4),
    ("tanh g=0.4 cap8",               8 * STEP,          "tanh", 0.4),
    ("tanh g=0.25 cap8",              8 * STEP,          "tanh", 0.25),
    ("clip g=0.33 cap4",              4 * STEP,          "clip", 0.33),
    ("clip g=0.33 cap8",              8 * STEP,          "clip", 0.33),
    ("linear g=0.33 cap4",            4 * STEP,          "linear", 0.33),
]
VARIANTS[0] = ("v3.5 shipped tanh g=1.0 cap4", 4 * STEP, "tanh", 1.0)


def main() -> None:
    rng = np.random.default_rng(SEED)
    judge_cache = {tuple(k.split("|", 1)): (float(v[0]), float(v[1]))
                   for k, v in json.loads(JUDGE_CACHE.read_text()).items()}
    data = load_datasets(judge_cache)
    print(f"loaded {len(data)} datasets\n")

    regimes = {}
    for base in sorted({d["base"] for d in data}):
        group = [d for d in data if d["base"] == base]
        if len(group) < 3:
            continue
        for test in group:
            train = [d for d in group if d["name"] != test["name"]]
            y = test["y"]
            if len(np.unique(y)) < 2:
                continue
            xt = np.concatenate([d["x"] for d in train])
            yt = np.concatenate([d["y"] for d in train])
            mu, sd = xt.mean(0), xt.std(0) + 1e-6
            model = LogisticRegression(max_iter=2000, C=0.01, random_state=SEED)
            model.fit((xt - mu) / sd, yt)
            probe = model.decision_function((test["x"] - mu) / sd)

            judge_z = zscore(test["judge"]) * (np.std(test["judge"]) /
                                               max(JUDGE_MARGIN_SD, 1e-9) + 1.0) / 1.0
            judge_z = zscore(test["judge"])
            j_auc = roc_auc_score(y, judge_z)

            for regime in ("natural", "iris_like", "notus"):
                if regime == "natural":
                    pz = zscore(probe)
                elif regime == "iris_like":
                    pz = retune_auroc(probe, y, min(j_auc + 0.068, 0.999), rng)
                else:
                    pz = retune_auroc(probe, y, 0.56, rng, sharpen_ok=False)
                regimes.setdefault(regime, []).append(
                    dict(base=base, organism=test["organism"], judge_z=judge_z,
                         probe_z=pz, y=y, j_auc=j_auc,
                         p_auc=roc_auc_score(y, pz)))

    print(f"{'variant':32s} " + "".join(f"{r:>12s}" for r in
                                        ("natural", "iris_like", "notus")) + f"{'headline':>12s}")
    rows = {}
    for label, max_cap, squash, gain in VARIANTS:
        cells = {}
        for regime, folds in regimes.items():
            aucs = [roc_auc_score(f["y"], fuse(f["judge_z"], f["probe_z"], max_cap, squash, gain))
                    for f in folds]
            cells[regime] = float(np.mean(aucs))
        headline = (cells["iris_like"] + cells["notus"]) / 2
        rows[label] = dict(cells=cells, headline=headline,
                           max_cap=max_cap, squash=squash, gain=gain)
        print(f"{label:32s} " + "".join(f"{cells[r]:12.4f}" for r in
                                        ("natural", "iris_like", "notus")) + f"{headline:12.4f}")

    ref = rows["v3.5 shipped tanh g=1.0 cap4"]["headline"]
    print(f"\ndeltas vs shipped v3.5 (headline = mean of iris_like and notus):")
    for label, r in sorted(rows.items(), key=lambda kv: -kv[1]["headline"]):
        print(f"  {label:32s} {r['headline'] - ref:+.4f}")

    print("\n--- saturation diagnostic ---")
    for regime in ("iris_like", "notus"):
        frac = np.mean([np.mean(np.abs(f["probe_z"]) > 2.0) for f in regimes[regime]])
        print(f"{regime:10s}: {frac*100:5.1f}% of rows have |probe_z| > 2 "
              f"(tanh within 4% of asymptote)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({k: {kk: vv for kk, vv in v.items()}
                                    for k, v in rows.items()}, indent=1))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
