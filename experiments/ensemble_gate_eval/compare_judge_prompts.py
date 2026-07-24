"""Compare two judge system prompts on identical rows.

This isolates sonic_v3 change 1. Both arms come from `run_judge_local.py` with
the same model, adapter, precision, rows and batching -- only the system prompt
differs -- so any movement here is the prompt and nothing else.

Read the output as a DOWNSIDE BOUND, not as a selection metric. Dev is
instructed/varied deception, which is the data that earned the narrow
"predict 1 only when the final answer itself is false" clause. A wider prompt is
expected to cost something here. The question is how much, and whether the
positive rate moves enough to disturb the gate's threshold.

Three numbers matter, in this order:

  positive_rate  the fraction of rows the judge calls deceptive. The gate sets
                 its threshold at quantile(score, 1 - positive_rate), so this
                 drives the balanced-accuracy column directly. Dev is 50%
                 deceptive, so a rate far from 0.50 is a threshold problem.
  soft_auroc     ranking quality of the soft confidence. This drives the AUROC
                 column through the blend.
  hard_ba        balanced accuracy of the hard verdicts on their own.

    python experiments/ensemble_gate_eval/compare_judge_prompts.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MANIFEST = REPO_ROOT / "dev_splits/manifest.csv"


def load_labels() -> dict:
    with open(MANIFEST) as handle:
        return {(row["dataset"], row["index"]): row["deceptive"] == "True"
                for row in csv.DictReader(handle)}


def metrics(hard: np.ndarray, soft: np.ndarray, labels: np.ndarray) -> dict:
    both = len(np.unique(labels)) > 1
    predicted = hard.astype(bool)
    true_positive = int((predicted & labels).sum())
    false_positive = int((predicted & ~labels).sum())
    return {
        "n": int(len(labels)),
        "positive_rate": float(predicted.mean()),
        "soft_auroc": float(roc_auc_score(labels, soft)) if both else float("nan"),
        "hard_ba": float(balanced_accuracy_score(labels, predicted)),
        "recall": true_positive / max(1, int(labels.sum())),
        "fpr": false_positive / max(1, int((~labels).sum())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2", type=Path, default=HERE / "judge_ab_v2.json")
    parser.add_argument("--v3", type=Path, default=HERE / "judge_ab_v3.json")
    args = parser.parse_args(argv)

    labels_by_key = load_labels()
    arms = {}
    for name, path in (("v2", args.v2), ("v3", args.v3)):
        raw = json.loads(path.read_text())
        arms[name] = {tuple(key.split("|", 1)): value for key, value in raw.items()}

    shared = sorted(set(arms["v2"]) & set(arms["v3"]) & set(labels_by_key))
    if not shared:
        raise SystemExit("no rows shared between the two arms and the manifest")
    print(f"comparing {len(shared)} rows present in both arms\n")

    datasets = sorted({key[0] for key in shared})
    header = f"{'dataset':<52}{'arm':>4}{'rate':>8}{'AUROC':>8}{'BA':>8}{'recall':>8}{'FPR':>7}"
    print(header)
    print("-" * len(header))

    overall = {}
    for name in ("v2", "v3"):
        overall[name] = metrics(
            np.array([arms[name][k][0] for k in shared]),
            np.array([arms[name][k][1] for k in shared]),
            np.array([labels_by_key[k] for k in shared]))

    for dataset in datasets:
        keys = [k for k in shared if k[0] == dataset]
        labels = np.array([labels_by_key[k] for k in keys])
        for name in ("v2", "v3"):
            result = metrics(np.array([arms[name][k][0] for k in keys]),
                             np.array([arms[name][k][1] for k in keys]), labels)
            label = dataset.replace("aletheias-quest/dev-", "")[:50] if name == "v2" else ""
            print(f"{label:<52}{name:>4}{result['positive_rate']:>8.3f}"
                  f"{result['soft_auroc']:>8.4f}{result['hard_ba']:>8.4f}"
                  f"{result['recall']:>8.3f}{result['fpr']:>7.3f}")
        print()

    print("=" * len(header))
    for name in ("v2", "v3"):
        result = overall[name]
        print(f"{'OVERALL':<52}{name:>4}{result['positive_rate']:>8.3f}"
              f"{result['soft_auroc']:>8.4f}{result['hard_ba']:>8.4f}"
              f"{result['recall']:>8.3f}{result['fpr']:>7.3f}")
    delta = {key: overall["v3"][key] - overall["v2"][key]
             for key in ("positive_rate", "soft_auroc", "hard_ba", "recall", "fpr")}
    print(f"{'CHANGE (v3 - v2)':<52}{'':>4}{delta['positive_rate']:>+8.3f}"
          f"{delta['soft_auroc']:>+8.4f}{delta['hard_ba']:>+8.4f}"
          f"{delta['recall']:>+8.3f}{delta['fpr']:>+7.3f}")

    disagreements = sum(1 for k in shared if arms["v2"][k][0] != arms["v3"][k][0])
    print(f"\nhard verdicts changed on {disagreements} of {len(shared)} rows "
          f"({disagreements / len(shared):.1%})")
    print("\nDev is the data that earned the narrow clause. Read a small loss here "
          "as the price of\nwider coverage on Notus, and a large one as a reason "
          "not to ship change 1.")


if __name__ == "__main__":
    main()
