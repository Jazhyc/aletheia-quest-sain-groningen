#!/usr/bin/env python3
"""Fit the sonic v3 probe/judge blend weight, leakage-free, by per-dataset AUROC.

sonic v3 drops the adaptive gate and scores every row on its own:

    score = sigmoid((1 - lambda) * probe_logit / sd_probe
                    + lambda * judge_margin / sd_judge)

Nothing in that expression reads the batch: ``lambda`` and both ``sd`` constants
are frozen here, offline.  Dividing by each detector's own spread is what makes
``lambda`` mean the same thing for the refit probe used here and the shipped
probe used at inference; the detector *means* are deliberately absent because an
additive constant shared by every row of a dataset cannot change its AUROC.

The script also reports what a ``has_lora``-keyed ``lambda`` would buy, so the
decision to ship a single weight stays measured rather than assumed.

Probe scores are refit leave-one-organism-out from cached activations, so a
held-out dataset never contributed to the probe that scores it.  Judge scores
come from ``run_direct_judge_local.py``.

    python experiments/ensemble_gate_eval/fit_direct_blend.py
    python experiments/ensemble_gate_eval/fit_direct_blend.py --max-epochs 8 \
        --base-models qwen
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "experiments/nonlinear_probe"))

from run_eval import (  # noqa: E402
    BaseModelData,
    load_manifest_rows,
    loo_folds,
)
from token_probes import TokenProbe, build_token_probe_model  # noqa: E402

DIRECT_JUDGE_CACHE = HERE / "direct_judge_cache.json"
SUBMISSION_PROBES = REPO_ROOT / "submission/whitebox_probe"
OUT_JSON = REPO_ROOT / "results/ensemble_gate_eval/direct_blend_fit.json"
PROBE_DIR = {"gemma": "gemma_probe", "qwen": "qwen_probe", "nemotron": "nemotron_probe"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
LAMBDA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)


def load_direct_judge_cache(path: Path) -> dict[tuple[str, str], float]:
    """Return ``(dataset, index) -> logit_1 - logit_0`` for the direct judge."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run run_direct_judge_local.py first"
        )
    raw = json.loads(path.read_text())
    margins = {}
    for key, (logit_zero, logit_one) in raw.items():
        dataset, index = key.split("|", 1)
        margins[(dataset, index)] = float(logit_one) - float(logit_zero)
    return margins


def refit_and_score_logits(data: BaseModelData, train_rows: np.ndarray,
                           test_rows: np.ndarray, max_epochs: int):
    """Refit the probe on train rows only and return held-out raw log-odds.

    Same recipe as ``run_eval.refit_and_score``, but reading the pre-sigmoid
    score so confident rows stay separable.
    """
    train_flat, train_offsets, train_labels, _ = data.subset(train_rows)
    probe = TokenProbe("transformer", seed=SEED, device=DEVICE,
                       max_epochs=max_epochs).fit(train_flat, train_offsets,
                                                  train_labels)
    test_flat, test_offsets, test_labels, test_keys = data.subset(test_rows)
    return probe.decision_function(test_flat, test_offsets), test_labels, test_keys


def load_shipped_probe(model: str, hidden_dim: int) -> TokenProbe:
    """Rebuild the exported submission probe so its score spread can be frozen."""
    probe_dir = SUBMISSION_PROBES / PROBE_DIR[model]
    config = json.loads((probe_dir / "config.json").read_text())
    network = build_token_probe_model(config["architecture"], hidden_dim).to(DEVICE)
    network.load_state_dict(torch.load(probe_dir / "model.pt", map_location=DEVICE))
    network.eval()
    probe = TokenProbe(config["architecture"], seed=SEED, device=DEVICE,
                       batch_token_budget=config.get("batch_token_budget", 8192))
    probe.model = network
    probe.feature_mean = torch.load(probe_dir / "feature_mean.pt", map_location=DEVICE)
    probe.feature_std = torch.load(probe_dir / "feature_std.pt", map_location=DEVICE)
    return probe


def collect_folds(models: list[str], manifest: list[dict], margins: dict,
                  max_epochs: int) -> tuple[list[dict], dict[str, float]]:
    """Refit leave-one-organism-out and pair each held-out row with the judge.

    :return: (per-organism fold records, shipped-probe logit spread per family).
    """
    folds: list[dict] = []
    shipped_spread: dict[str, float] = {}
    for model in models:
        data = BaseModelData(model, manifest)
        organisms = sorted(set(data.organisms.tolist()))
        print(f"\n{model}: {len(data.labels)} rows, {len(organisms)} organisms")

        shipped = load_shipped_probe(model, data.flat.shape[1])
        all_rows = np.arange(len(data.labels))
        shipped_logit = np.concatenate([
            shipped.decision_function(*data.subset(all_rows[data.organisms == organism])[:2])
            for organism in organisms
        ])
        shipped_spread[model] = float(np.std(shipped_logit))
        print(f"  shipped probe logit sd = {shipped_spread[model]:.4f}")

        if len(organisms) < 2:
            print("  only one organism: no leakage-free refit possible, skipped")
            continue

        for held, train_rows, test_rows in loo_folds(data, None):
            scores, labels, keys = refit_and_score_logits(data, train_rows,
                                                          test_rows, max_epochs)
            paired = [(probe, margins[(dataset, index)], label)
                      for probe, (dataset, index), label
                      in zip(scores, keys, labels)
                      if (dataset, index) in margins]
            if len(paired) < len(keys):
                print(f"  {held}: judge cache missing "
                      f"{len(keys) - len(paired)}/{len(keys)} rows")
            if not paired:
                continue
            probe_scores, judge_margins, fold_labels = map(np.asarray, zip(*paired))
            folds.append({
                "model": model,
                "organism": held,
                "dataset": keys[0][0],
                "has_lora": not held.endswith("/base"),
                "probe_logit": probe_scores.astype(float),
                "judge_margin": judge_margins.astype(float),
                "labels": fold_labels.astype(int),
            })
            print(f"  {held}: {len(paired)} rows")
    return folds, shipped_spread


def blend_auroc(fold: dict, weight: float, probe_scale: float,
                judge_scale: float) -> float:
    """AUROC of one held-out dataset under a fixed blend weight."""
    combined = ((1.0 - weight) * fold["probe_logit"] / probe_scale
                + weight * fold["judge_margin"] / judge_scale)
    if len(np.unique(fold["labels"])) < 2:
        return float("nan")
    return float(roc_auc_score(fold["labels"], combined))


def sweep(folds: list[dict], probe_scales: dict[str, float],
          judge_scale: float) -> dict[float, dict[str, list]]:
    """Per-lambda, per-regime AUROC across every held-out dataset."""
    table: dict[float, dict[str, list]] = {}
    for weight in LAMBDA_GRID:
        by_regime: dict[str, list] = defaultdict(list)
        for fold in folds:
            auroc = blend_auroc(fold, float(weight), probe_scales[fold["model"]],
                                judge_scale)
            if not np.isnan(auroc):
                # Key on model AND organism: "instr/base" exists for more than
                # one base model, and collapsing them loses a fold.
                name = f"{fold['model']}:{fold['organism']}"
                by_regime["all"].append((name, auroc))
                by_regime["lora" if fold["has_lora"] else "base"].append(
                    (name, auroc))
        table[float(weight)] = dict(by_regime)
    return table


def summarize(entries: list[tuple[str, float]]) -> dict[str, float]:
    values = [auroc for _, auroc in entries]
    return {"mean": float(np.mean(values)), "min": float(np.min(values)),
            "n": len(values)}


def report_regime(table: dict, regime: str) -> dict:
    """Pick the mean-optimal lambda and record whether it dominates the ends."""
    rows = {weight: summarize(by_regime[regime])
            for weight, by_regime in table.items() if regime in by_regime}
    if not rows:
        return {}
    best = max(rows, key=lambda weight: rows[weight]["mean"])
    print(f"\n--- regime: {regime} ({rows[best]['n']} datasets) ---")
    for weight in sorted(rows):
        marker = "  <-- best" if weight == best else ""
        print(f"  lambda={weight:.2f}  mean AUROC={rows[weight]['mean']:.4f}  "
              f"worst={rows[weight]['min']:.4f}{marker}")
    per_dataset = {organism: auroc for organism, auroc in table[best][regime]}
    beats_probe = rows[best]["mean"] - rows[0.0]["mean"]
    beats_judge = rows[best]["mean"] - rows[1.0]["mean"]
    print(f"  gain over probe-only: {beats_probe:+.4f}   "
          f"over judge-only: {beats_judge:+.4f}")
    return {
        "best_lambda": best,
        "grid": {str(weight): rows[weight] for weight in sorted(rows)},
        "per_dataset_at_best": per_dataset,
        "gain_over_probe_only": beats_probe,
        "gain_over_judge_only": beats_judge,
    }


def fit_threshold(folds: list[dict], weights: dict[bool, float],
                  probe_scales: dict[str, float], judge_scale: float,
                  use_probe: bool, use_judge: bool) -> float:
    """Frozen cut maximizing mean per-dataset balanced accuracy for one path.

    The binary column no longer scores -- the leaderboard ranks on AUROC -- so
    this exists only to keep the displayed balanced accuracy sane.  It is a
    constant chosen offline, never a function of the scored batch.
    """
    from sklearn.metrics import balanced_accuracy_score

    per_fold = []
    for fold in folds:
        weight = weights[fold["has_lora"]] if (use_probe and use_judge) else (
            1.0 if use_judge else 0.0)
        combined = ((1.0 - weight) * fold["probe_logit"] / probe_scales[fold["model"]]
                    + weight * fold["judge_margin"] / judge_scale)
        per_fold.append((1.0 / (1.0 + np.exp(-np.clip(combined, -80.0, 80.0))),
                         fold["labels"]))
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 3)
    means = [np.mean([balanced_accuracy_score(labels, scores >= cut)
                      for scores, labels in per_fold])
             for cut in candidates]
    best = float(candidates[int(np.argmax(means))])
    print(f"  threshold={best:.2f} mean BA={max(means):.4f} "
          f"(probe={use_probe} judge={use_judge})")
    return best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-models", nargs="+",
                        default=["qwen", "gemma", "nemotron"])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--judge-cache", type=Path, default=DIRECT_JUDGE_CACHE)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    margins = load_direct_judge_cache(args.judge_cache)
    print(f"direct judge cache: {len(margins)} rows")
    judge_scale = float(np.std(list(margins.values())))
    print(f"judge margin sd = {judge_scale:.4f}")

    manifest = load_manifest_rows()
    folds, shipped_spread = collect_folds(args.base_models, manifest, margins,
                                          args.max_epochs)
    if not folds:
        raise SystemExit("no leakage-free folds produced")

    refit_scales = {}
    for model in {fold["model"] for fold in folds}:
        pooled = np.concatenate([fold["probe_logit"] for fold in folds
                                 if fold["model"] == model])
        refit_scales[model] = float(np.std(pooled))
        print(f"refit probe logit sd ({model}) = {refit_scales[model]:.4f}")

    table = sweep(folds, refit_scales, judge_scale)
    regimes = {name: report_regime(table, name) for name in ("all", "base", "lora")}

    # One weight for every row. The per-regime fit was measured and rejected:
    # keying lambda on has_lora moved the overall mean by less than 1e-4, which
    # is two free parameters buying nothing. See the regime tables above.
    global_weight = regimes["all"]["best_lambda"]
    weights = {False: global_weight, True: global_weight}
    print("\n--- frozen thresholds (balanced accuracy only) ---")
    constants = {
        "lambda_blend": global_weight,
        "lambda_base_if_split": regimes["base"]["best_lambda"],
        "lambda_lora_if_split": regimes["lora"]["best_lambda"],
        "probe_logit_sd": shipped_spread,
        "default_probe_sd": float(np.mean(list(shipped_spread.values()))),
        "judge_margin_sd": judge_scale,
        "threshold_blend": fit_threshold(folds, weights, refit_scales, judge_scale,
                                         use_probe=True, use_judge=True),
        "threshold_probe_only": fit_threshold(folds, weights, refit_scales, judge_scale,
                                              use_probe=True, use_judge=False),
        "threshold_judge_only": fit_threshold(folds, weights, refit_scales, judge_scale,
                                              use_probe=False, use_judge=True),
    }
    constants_path = args.out.parent / "sonic_v3_constants.json"
    constants_path.parent.mkdir(parents=True, exist_ok=True)
    constants_path.write_text(json.dumps(constants, indent=2) + "\n")
    print(f"\nwrote {constants_path}")

    folds_npz = args.out.parent / "direct_blend_folds.npz"
    np.savez_compressed(folds_npz, **{
        f"{fold['model']}:{fold['organism']}|{field}": fold[field]
        for fold in folds
        for field in ("probe_logit", "judge_margin", "labels")
    })
    print(f"wrote {folds_npz}")

    report = {
        "constants": constants,
        "per_dataset_grid": {
            str(weight): dict(by_regime.get("all", []))
            for weight, by_regime in table.items()
        },
        "judge_margin_sd": judge_scale,
        "refit_probe_logit_sd": refit_scales,
        "shipped_probe_logit_sd": shipped_spread,
        "regimes": regimes,
        "folds": [{"organism": fold["organism"], "model": fold["model"],
                   "dataset": fold["dataset"], "has_lora": fold["has_lora"],
                   "rows": int(len(fold["labels"])),
                   "deceptive": int(fold["labels"].sum())}
                  for fold in folds],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
