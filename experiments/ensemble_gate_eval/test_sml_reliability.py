"""Does a Parisi-style spectral estimator recover probe reliability without labels?

The sonic gate currently guesses probe reliability with a frozen constant. The
proposal is to *estimate* it per batch from detector agreement alone, using the
Spectral Meta-Learner of Parisi, Strino, Nadler & Kluger (PNAS 2014): under
conditional independence given the label, the off-diagonal of the detector
covariance matrix is rank one, `q_ij = v_i v_j`, and `v_i` is proportional to
detector `i`'s Youden index `2*BA - 1`.

Three detectors are required for identifiability. With `M = 3` the system is
exactly determined and has the closed form `v_1^2 = q_12 q_13 / q_23`.

The three detectors here are the judge, a probe on layer 46, and a probe on
layer 40. Layer 40 is known to share errors with layer 46 -- both read the same
residual stream through the same trunk -- which violates conditional
independence. Measuring how badly is the point of this script.

Detectors are binarised at their **batch median**, not at a shipped threshold.
The competition metric is AUROC and `sonic_v3_3_mini` produced Iris/gemma at
BA 0.5000 / AUROC 0.9856, so a threshold-based binarisation inverts exactly the
case the redesign exists to rescue.

    python experiments/ensemble_gate_eval/test_sml_reliability.py
    python experiments/ensemble_gate_eval/test_sml_reliability.py --blunt 0.56
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ACTIVATIONS = REPO_ROOT / "results/whitebox/activations"
JUDGE_CACHE = HERE / "judge_cache.json"
OUT_JSON = REPO_ROOT / "results/ensemble_gate_eval/sml_reliability.json"

SEED = 0
LAYER_MAIN = "mean_L46"
LAYER_THIRD = "mean_L40"


def base_model_of(name: str) -> str:
    if "gemma-3-27b-it" in name:
        return "gemma"
    if "Qwen3.5-27B" in name:
        return "qwen"
    if "Nemotron" in name:
        return "nemotron"
    return "unknown"


def organism_of(name: str) -> str:
    """Everything after the base-model tag -- the LoRA id, or `None` for base."""
    for tag in ("gemma-3-27b-it-", "Qwen3.5-27B-", "A12B-BF16-"):
        if tag in name:
            return name.split(tag, 1)[1]
    return "?"


def load_datasets(judge_cache: dict) -> list[dict]:
    """Load every full activation cache that has judge coverage."""
    out = []
    for path in sorted(ACTIVATIONS.glob("*.npz")):
        stem = path.name[: -len(".npz")]
        if ".tokens" in stem or ".limit" in stem:
            continue
        blob = np.load(path, allow_pickle=True)
        if LAYER_MAIN not in blob.files or LAYER_THIRD not in blob.files:
            continue
        ds_key = stem.replace("__", "/")
        index = blob["index"]
        judge = np.array(
            [judge_cache.get((ds_key, str(int(i))), (np.nan, np.nan))[1] for i in index],
            dtype=np.float64,
        )
        if not np.isfinite(judge).all():
            continue
        out.append(
            dict(
                name=ds_key,
                base=base_model_of(stem),
                organism=organism_of(stem),
                x46=blob[LAYER_MAIN].astype(np.float32),
                x40=blob[LAYER_THIRD].astype(np.float32),
                y=blob["deceptive"].astype(int),
                judge=judge,
            )
        )
    return out


def fit_loo_probe(train: list[dict], test: dict, key: str) -> np.ndarray:
    """Leave-one-organism-out logistic probe; returns decision scores on `test`."""
    x_train = np.concatenate([d[key] for d in train])
    y_train = np.concatenate([d["y"] for d in train])
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    model = LogisticRegression(max_iter=2000, C=0.01, random_state=SEED)
    model.fit((x_train - mu) / sd, y_train)
    return model.decision_function((test[key] - mu) / sd)


def blunt(scores: np.ndarray, labels: np.ndarray, target_auroc: float,
          rng: np.random.Generator) -> np.ndarray:
    """Add noise until the score's AUROC lands near `target_auroc`."""
    z = (scores - scores.mean()) / (scores.std() + 1e-9)
    lo, hi = 0.0, 200.0
    for _ in range(40):
        mid = (lo + hi) / 2
        noisy = z + mid * rng.standard_normal(z.shape)
        got = roc_auc_score(labels, noisy)
        if got > target_auroc:
            lo = mid
        else:
            hi = mid
    return z + ((lo + hi) / 2) * rng.standard_normal(z.shape)


def sml_weights(binarised: np.ndarray) -> np.ndarray:
    """Parisi et al. closed form for M=3: v_i^2 = q_ij q_ik / q_jk.

    :param binarised: (3, n) array of +/-1 detector predictions.
    :returns: length-3 vector proportional to each detector's Youden index.
    """
    q = np.cov(binarised)
    v = np.zeros(3)
    for i, (j, k) in enumerate([(1, 2), (0, 2), (0, 1)]):
        num = q[i, j] * q[i, k]
        den = q[j, k]
        v[i] = np.sqrt(max(num / den, 0.0)) if abs(den) > 1e-12 and num / den > 0 else 0.0
    # Sign: a detector anti-correlated with the other two is inverted.
    for i, (j, k) in enumerate([(1, 2), (0, 2), (0, 1)]):
        if q[i, j] + q[i, k] < 0:
            v[i] = -v[i]
    return v


def median_binarise(scores: np.ndarray) -> np.ndarray:
    signs = np.sign(scores - np.median(scores))
    signs[signs == 0] = 1.0
    return signs


def youden(scores: np.ndarray, labels: np.ndarray) -> float:
    return 2 * balanced_accuracy_score(labels, (median_binarise(scores) > 0).astype(int)) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blunt", type=float, default=None,
                        help="Blunt both probes to this AUROC (simulates Notus).")
    parser.add_argument("--blunt-corr", type=float, default=1.0,
                        help="1.0 = probes fail with identical noise, 0.0 = independent.")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    judge_cache = {tuple(k.split("|", 1)): (float(v[0]), float(v[1]))
                   for k, v in json.loads(JUDGE_CACHE.read_text()).items()}
    data = load_datasets(judge_cache)
    print(f"loaded {len(data)} datasets with judge coverage")

    rows = []
    for base in sorted({d["base"] for d in data}):
        group = [d for d in data if d["base"] == base]
        if len(group) < 3:
            continue
        for test in group:
            train = [d for d in group if d["name"] != test["name"]]
            y = test["y"]
            if len(np.unique(y)) < 2:
                continue
            p46 = fit_loo_probe(train, test, "x46")
            p40 = fit_loo_probe(train, test, "x40")
            if args.blunt is not None:
                shared = rng.standard_normal(p46.shape)
                n46 = blunt(p46, y, args.blunt, rng)
                n40_ind = blunt(p40, y, args.blunt, rng)
                zz = (p40 - p40.mean()) / (p40.std() + 1e-9)
                n40 = args.blunt_corr * (n46 - (p46 - p46.mean()) / (p46.std() + 1e-9) + zz) \
                    + (1 - args.blunt_corr) * n40_ind
                p46, p40 = n46, n40
                del shared
            det = np.vstack([test["judge"], p46, p40])
            binar = np.vstack([median_binarise(r) for r in det])
            v = sml_weights(binar)
            true_auroc = [roc_auc_score(y, r) for r in det]
            true_youden = [youden(r, y) for r in det]
            rows.append(dict(base=base, organism=test["organism"], n=int(len(y)),
                             v=[float(x) for x in v],
                             auroc=[float(x) for x in true_auroc],
                             youden=[float(x) for x in true_youden]))
            print(f"{base:9s} {test['organism'][:22]:22s} n={len(y):4d} "
                  f"v=[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}]  "
                  f"true_auroc=[{true_auroc[0]:.3f} {true_auroc[1]:.3f} {true_auroc[2]:.3f}]  "
                  f"true_youden=[{true_youden[0]:+.3f} {true_youden[1]:+.3f} {true_youden[2]:+.3f}]")

    if not rows:
        print("no folds produced")
        return

    v_all = np.array([r["v"] for r in rows])
    y_all = np.array([r["youden"] for r in rows])
    a_all = np.array([r["auroc"] for r in rows])
    names = ["judge", "probe_L46", "probe_L40"]

    print("\n=== estimator quality (across folds, per detector) ===")
    for i, nm in enumerate(names):
        c_y = np.corrcoef(v_all[:, i], y_all[:, i])[0, 1]
        c_a = np.corrcoef(v_all[:, i], a_all[:, i])[0, 1]
        print(f"{nm:10s} corr(v, true_youden)={c_y:+.3f}  corr(v, true_auroc)={c_a:+.3f}  "
              f"mean_v={v_all[:, i].mean():+.3f}  mean_auroc={a_all[:, i].mean():.3f}")

    print("\n=== the question that decides it ===")
    est_ratio = v_all[:, 1] / np.maximum(np.abs(v_all[:, 0]), 1e-9)
    true_ratio = y_all[:, 1] / np.maximum(np.abs(y_all[:, 0]), 1e-9)
    print(f"corr(estimated probe/judge ratio, true ratio) = "
          f"{np.corrcoef(est_ratio, true_ratio)[0, 1]:+.3f}")
    print(f"mean estimated probe/judge ratio = {est_ratio.mean():+.3f}   "
          f"(true {true_ratio.mean():+.3f})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(blunt=args.blunt, rows=rows), indent=1))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
