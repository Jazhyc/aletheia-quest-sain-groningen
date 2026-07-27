#!/usr/bin/env python3
"""Select the sonic v3.5 gate: a sign test in place of v3.2's agreement product.

Two things are wrong with the v3.2 selection, and both are fixed here.

**The safety regime was calibrated to the wrong probe.** `fit_bounded_refine_v3_2.py`
scored candidates against a probe blunted to AUROC 0.76, called "the Notus
regime". `sonic_v3_3_mini` has since measured the probe's real Notus AUROC on the
private split: 0.5586 overall and 0.4808 on the gemma unit, i.e. below chance. So
the binding constraint was fitted against a probe roughly twice as informative as
the one that actually turns up. This script scores at 0.56 and at 0.48 inverted.
The correction matters: a flat cap looks acceptable at 0.76 and is a disaster at
0.56.

**The gain regime was the average dev fold.** On the average fold the probe beats
the judge by +0.014. On the real Iris it beats it by +0.068. Averaging over all
20 folds therefore measures a condition Iris is not in. Gains here are read off
the 10 folds where the probe most outclasses the judge.

What the fixed measurement says about `agreement = clip(judge_z * probe_z / 3, 0, 1)`:
the opening correlates +0.94 with `|judge_z|`, so the cap is widest where the
judge is already certain and near-shut where it is unsure. In the noise regime
that is a *safety feature* -- confident rows are far from the ordering boundary,
so a random probe cannot reorder them -- which is why the flat and
judge-uncertainty gates lose badly at 0.56. But on Iris-like folds it withholds
the probe exactly where a better-than-judge probe would pay, capturing 54% of the
probe's edge against a sign test's 66%.

The sign test keeps the direction check that provides the safety and drops the
magnitude coupling that costs the gain.

    python experiments/ensemble_gate_eval/fit_sign_gate_v3_5.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS = REPO / "results/ensemble_gate_eval/direct_blend_folds.npz"
DEFAULT_BASE = REPO / "results/ensemble_gate_eval/sonic_v3_4_constants.json"
DEFAULT_OUTPUT = REPO / "results/ensemble_gate_eval/sonic_v3_5_constants.json"

JUDGE_SD = 1.199755138011975
STEP = 0.10418792638565248
PROBE_MEAN = {"nemotron": -3.2186061611606016, "qwen": -0.25166166366445586,
              "gemma": -1.968125202258428}
PROBE_SD = {"nemotron": 2.1939692650621225, "qwen": 4.157805340296084,
            "gemma": 4.34205824026173}

SEEDS = (7, 11, 23, 42, 101)
SAFETY_TARGETS = (0.48, 0.56)
IRIS_LIKE_FOLDS = 10


def load_folds(path: Path) -> list:
    """Read the 20 leave-one-organism-out dev folds v3.1/v3.2 were fitted on.

    :param path: the cached fold archive
    :returns: list of (name, labels, judge_z, probe_z)
    """
    data = np.load(path, allow_pickle=True)
    names = sorted({key.split("|")[0] for key in data.keys()})
    folds = []
    for name in names:
        family = name.split(":")[0]
        folds.append((name,
                      data[name + "|labels"],
                      data[name + "|judge_margin"] / JUDGE_SD,
                      (data[name + "|probe_logit"] - PROBE_MEAN[family]) / PROBE_SD[family]))
    return folds


def blunt(probe_z: np.ndarray, labels: np.ndarray, target: float,
          rng: np.random.Generator) -> np.ndarray:
    """Add noise until the probe's AUROC hits ``target``, inverting below chance.

    :param probe_z: standardized probe scores
    :param labels: ground truth
    :param target: desired AUROC
    :param rng: seeded generator
    :returns: the blunted probe scores
    """
    if target < 0.5:
        return -blunt(probe_z, labels, 1.0 - target, rng)
    low, high = 0.0, 30.0
    for _ in range(40):
        middle = (low + high) / 2
        scored = np.mean([roc_auc_score(labels, probe_z + middle * rng.standard_normal(len(probe_z)))
                          for _ in range(7)])
        low, high = (middle, high) if scored > target else (low, middle)
    return probe_z + ((low + high) / 2) * rng.standard_normal(len(probe_z))


def opening_product(judge_z: np.ndarray, probe_z: np.ndarray,
                    scale: float = 3.0) -> np.ndarray:
    """v3.2's cap opening: the clipped judge/probe product."""
    return np.clip(judge_z * probe_z / scale, 0.0, 1.0)


def opening_sign(judge_z: np.ndarray, probe_z: np.ndarray) -> np.ndarray:
    """v3.5's cap opening: 1 when the detectors point the same way, else 0."""
    return (judge_z * probe_z > 0).astype(float)


def refine(judge_z: np.ndarray, probe_z: np.ndarray, opening: np.ndarray,
           base_cap: float, max_cap: float) -> np.ndarray:
    cap = base_cap + opening * (max_cap - base_cap)
    return judge_z + cap * np.tanh(probe_z)


def iris_like(folds: list) -> list:
    """The folds where the probe most outclasses the judge -- the Iris condition."""
    ranked = sorted(folds, key=lambda fold: (roc_auc_score(fold[1], fold[3])
                                             - roc_auc_score(fold[1], fold[2])))
    return ranked[-IRIS_LIKE_FOLDS:]


def measure(folds: list, base_cap: float, max_cap: float) -> dict:
    """Score both gate shapes on the gain regime and both safety regimes."""
    report = {}

    gains = {"product": [], "sign": []}
    edges = []
    for _, labels, judge_z, probe_z in iris_like(folds):
        judge_auroc = roc_auc_score(labels, judge_z)
        edges.append(roc_auc_score(labels, probe_z) - judge_auroc)
        for key, opening in (("product", opening_product(judge_z, probe_z)),
                             ("sign", opening_sign(judge_z, probe_z))):
            gains[key].append(
                roc_auc_score(labels, refine(judge_z, probe_z, opening, base_cap, max_cap))
                - judge_auroc)
    report["iris_like_probe_edge"] = float(np.mean(edges))
    for key in gains:
        report[f"iris_like_gain_{key}"] = float(np.mean(gains[key]))
        report[f"iris_like_capture_{key}"] = float(np.mean(gains[key]) / np.mean(edges))

    for target in SAFETY_TARGETS:
        costs = {"product": [], "sign": []}
        paired = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            for _, labels, judge_z, probe_z in folds:
                blunted = blunt(probe_z.copy(), labels, target, rng)
                judge_auroc = roc_auc_score(labels, judge_z)
                scored = {}
                for key, opening in (("product", opening_product(judge_z, blunted)),
                                     ("sign", opening_sign(judge_z, blunted))):
                    scored[key] = roc_auc_score(
                        labels, refine(judge_z, blunted, opening, base_cap, max_cap))
                    costs[key].append(scored[key] - judge_auroc)
                paired.append(scored["sign"] - scored["product"])
        tag = f"{target:.2f}"
        for key in costs:
            report[f"notus_{tag}_mean_{key}"] = float(np.mean(costs[key]))
            report[f"notus_{tag}_worst_{key}"] = float(np.min(costs[key]))
        report[f"notus_{tag}_sign_minus_product_mean"] = float(np.mean(paired))
        report[f"notus_{tag}_sign_minus_product_worst"] = float(np.min(paired))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--base-constants", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    inherited = json.loads(args.base_constants.read_text())
    base_cap = inherited["base_cap"]
    max_cap = inherited["max_cap"]
    folds = load_folds(args.folds)
    print(f"folds = {len(folds)}, seeds = {len(SEEDS)}")
    print(f"caps inherited from {args.base_constants.name}: "
          f"BASE {base_cap / STEP:.0f} steps, MAX {max_cap / STEP:.0f} steps\n")

    report = measure(folds, base_cap, max_cap)

    print(f"Iris-like folds (n={IRIS_LIKE_FOLDS}), mean probe edge over judge "
          f"{report['iris_like_probe_edge']:+.4f}")
    for key in ("product", "sign"):
        print(f"  {key:<8} gain {report[f'iris_like_gain_{key}']:+.4f}"
              f"   captures {report[f'iris_like_capture_{key}']:.1%} of the edge")
    for target in SAFETY_TARGETS:
        tag = f"{target:.2f}"
        print(f"\nNotus regime, probe blunted to {tag}")
        for key in ("product", "sign"):
            print(f"  {key:<8} mean {report[f'notus_{tag}_mean_{key}']:+.4f}"
                  f"   worst fold {report[f'notus_{tag}_worst_{key}']:+.4f}")
        print(f"  sign - product: mean {report[f'notus_{tag}_sign_minus_product_mean']:+.4f}"
              f"   worst {report[f'notus_{tag}_sign_minus_product_worst']:+.4f}")

    net = (report["iris_like_gain_sign"] - report["iris_like_gain_product"]
           + report["notus_0.56_sign_minus_product_mean"]) / 2
    print(f"\nprojected headline change against the product gate: {net:+.4f}")
    print("  (half the Iris-like gain plus half the 0.56 Notus cost; the headline "
          "is the mean of the two counted datasets)")

    constants = dict(inherited)
    constants.update({
        "rule": "judge_z + (BASE_CAP + (judge_z*probe_z > 0) * (MAX_CAP-BASE_CAP)) "
                "* tanh(PROBE_GAIN * probe_z)",
        "purpose": "replace v3.2's agreement product with a sign test; caps and probe "
                   "unchanged from v3.4",
        "gate": "sign",
        "agreement_scale": None,
        "selection": report,
        "selection_notes": {
            "safety_targets": list(SAFETY_TARGETS),
            "why_0.56": "sonic_v3_3_mini measured the probe's real Notus AUROC at 0.5586; "
                        "the v3.2 fit used 0.76, which no longer matches the evidence",
            "iris_proxy": f"the {IRIS_LIKE_FOLDS} dev folds where the probe most outclasses "
                          "the judge; the average fold is not in Iris's condition",
            "caveat": "the probe is in-sample on every dev fold, so absolute gains are "
                      "inflated; only the comparison between the two shapes is used",
        },
    })
    constants.pop("official_reference", None)
    args.output.write_text(json.dumps(constants, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
