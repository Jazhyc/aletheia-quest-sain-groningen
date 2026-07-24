"""Leakage-free leave-one-organism-out comparison of the absolute probe, the
contrastive (difference) probe, and their rank fusion.

This is the go/no-go for sonic v3 change 2. The question it answers is NOT
"which probe is better on dev" -- both are strong there -- but "which probe
survives an organism it has never seen", because that is the axis on which the
shipped probe fails (Notus, base-model-only, AUROC 0.69-0.83 against a rank-1
competitor at ~0.90).

The headline number is therefore the WORST held-out organism, not the mean. A
probe that averages well while collapsing on one organism is exactly the probe
we already have.

Both probes are refit per fold on train rows only, so every reported score is
leakage-free. Both read the same rows, in the same order, from the same folds --
the only difference is the feature set:

  absolute    -- results/whitebox/activations/*.tokens.npz, tokens_L46
  contrastive -- results/contrastive/activations/*.npz, diff_flat
                 (L46 activations with the system prompt minus without it)

Rows are intersected on (dataset, index) so a row missing from either cache is
dropped from both.

    python experiments/contrastive_probe/run_loo_eval.py --base-models qwen \
        --limit-organisms 3 --max-epochs 4          # ~minutes, validates wiring
    python experiments/contrastive_probe/run_loo_eval.py                # full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments/ensemble_gate_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments/nonlinear_probe"))

from gate import rank01  # noqa: E402
from invariant_probe import InvariantTokenProbe  # noqa: E402
from run_eval import SCEN_SHORT, base_model_of, load_manifest_rows, probe_layer  # noqa: E402
from token_probes import TokenProbe  # noqa: E402

METHODS = ("absolute", "contrastive", "fused", "invariant")

WHITEBOX = REPO_ROOT / "results/whitebox/activations"
CONTRASTIVE = REPO_ROOT / "results/contrastive/activations"
OUT_JSON = REPO_ROOT / "results/contrastive/loo_eval.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = np.finfo(np.float16)


def whitebox_path(dataset: str) -> Path:
    return WHITEBOX / f"{dataset.replace('/', '__')}.tokens.npz"


def contrastive_path(dataset: str) -> Path:
    return CONTRASTIVE / f"{dataset.replace('/', '_')}.npz"


def _as_fp16(array: np.ndarray) -> np.ndarray:
    """Clamp into the fp16 range before casting, so a large-magnitude family
    (gemma diffs run ~300x qwen's) cannot silently become inf."""
    return np.clip(array, FP16.min, FP16.max).astype(np.float16)


class AlignedData:
    """Absolute and contrastive token features for one base model, over the rows
    present in BOTH caches, with per-row organism labels from the manifest."""

    def __init__(self, model: str, manifest: list[dict]):
        self.model = model
        self.layer = probe_layer(model)
        datasets = sorted({row["dataset"] for row in manifest
                           if base_model_of(row["dataset"]) == model})

        abs_spans, ctr_spans, labels, organisms, keys = [], [], [], [], []
        self.skipped = []
        for dataset in datasets:
            if not contrastive_path(dataset).exists():
                self.skipped.append(dataset)
                continue
            with np.load(whitebox_path(dataset), allow_pickle=True) as data:
                abs_tokens = data[f"tokens_L{self.layer}"]
                abs_offsets = data["token_offsets"].astype(np.int64)
                abs_index = data["index"]
                abs_deceptive = data["deceptive"].astype(int)
            with np.load(contrastive_path(dataset), allow_pickle=True) as data:
                ctr_tokens = data["diff_flat"]
                ctr_offsets = data["offsets"].astype(np.int64)
                ctr_index = data["indices"]
                ctr_deceptive = data["deceptive"].astype(int)

            abs_row_of = {int(v): i for i, v in enumerate(abs_index)}
            ctr_row_of = {int(v): i for i, v in enumerate(ctr_index)}
            meta = {str(row["index"]): row for row in manifest
                    if row["dataset"] == dataset}

            for index_value, abs_row in abs_row_of.items():
                ctr_row = ctr_row_of.get(index_value)
                info = meta.get(str(index_value))
                if ctr_row is None or info is None:
                    continue
                label = int(abs_deceptive[abs_row])
                if int(ctr_deceptive[ctr_row]) != label:
                    raise ValueError(f"label mismatch on {dataset}:{index_value}")
                abs_spans.append(abs_tokens[abs_offsets[abs_row]:abs_offsets[abs_row + 1]])
                ctr_spans.append(_as_fp16(ctr_tokens[ctr_offsets[ctr_row]:ctr_offsets[ctr_row + 1]]))
                labels.append(label)
                scenario = SCEN_SHORT.get(info["scenario"], info["scenario"])
                organisms.append(f"{scenario}/{info['lora_id'] or 'base'}")
                keys.append((dataset, str(index_value)))

        if not labels:
            raise ValueError(f"no rows with both caches for {model}")

        self.abs_flat = np.concatenate(abs_spans, axis=0)
        self.abs_offsets = np.cumsum([0] + [len(s) for s in abs_spans]).astype(np.int64)
        self.ctr_flat = np.concatenate(ctr_spans, axis=0)
        self.ctr_offsets = np.cumsum([0] + [len(s) for s in ctr_spans]).astype(np.int64)
        self.labels = np.asarray(labels)
        self.organisms = np.asarray(organisms, dtype=object)
        self.keys = keys

    def subset(self, feature_set: str, rows: np.ndarray):
        """Repack the given row ids of one feature set into (flat tensor on
        DEVICE, offsets) for a probe fit or scoring pass."""
        flat, offsets = ((self.abs_flat, self.abs_offsets) if feature_set == "absolute"
                         else (self.ctr_flat, self.ctr_offsets))
        pieces = [flat[offsets[r]:offsets[r + 1]] for r in np.asarray(rows)]
        packed = np.concatenate(pieces, axis=0)
        row_offsets = np.cumsum([0] + [len(p) for p in pieces]).astype(np.int64)
        # The gemma whitebox cache holds ~700 inf values (fp16 overflow at
        # extraction time); unclamped they propagate to NaN scores. run_eval.py
        # applies the identical clamp, so this keeps the two harnesses matched.
        finfo = torch.finfo(torch.float16)
        return torch.from_numpy(packed).clamp(finfo.min, finfo.max).to(DEVICE), row_offsets

    @property
    def hidden_dim(self) -> int:
        return int(self.abs_flat.shape[1])


def loo_folds(data: AlignedData, limit: int | None):
    organisms = sorted(set(data.organisms.tolist()))
    if len(organisms) < 2:
        return
    if limit:
        organisms = organisms[:limit]
    all_rows = np.arange(len(data.labels))
    for held in organisms:
        test_rows = all_rows[data.organisms == held]
        train_rows = all_rows[data.organisms != held]
        if limit:  # smoke: restrict the training pool to the kept organisms too
            keep = np.isin(data.organisms, organisms)
            train_rows = all_rows[keep & (data.organisms != held)]
        if len(np.unique(data.labels[train_rows])) < 2 or len(test_rows) == 0:
            continue
        yield held, train_rows, test_rows


def refit_and_score(data: AlignedData, feature_set: str, train_rows, test_rows,
                    max_epochs: int, seed: int, invariant: bool = False) -> np.ndarray:
    """Refit one probe on train_rows and score test_rows. Leakage-free: the
    probe only ever sees train rows (and its own validation split of them).

    :param invariant: Train the domain-adversarial variant, using the training
        organisms as adversary classes. The held-out organism is by construction
        absent from them, which is the point of the fold.
    """
    train_flat, train_offsets = data.subset(feature_set, train_rows)
    if invariant:
        probe = InvariantTokenProbe(seed=seed, device=DEVICE, max_epochs=max_epochs)
        probe.fit(train_flat, train_offsets, data.labels[train_rows],
                  data.organisms[train_rows])
    else:
        probe = TokenProbe("transformer", seed=seed, device=DEVICE,
                           max_epochs=max_epochs).fit(train_flat, train_offsets,
                                                      data.labels[train_rows])
    del train_flat
    test_flat, test_offsets = data.subset(feature_set, test_rows)
    scores = probe.predict_proba(test_flat, test_offsets)[:, 1]
    del test_flat
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return scores


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def run_model(data: AlignedData, max_epochs: int, seeds: list[int],
              limit: int | None, methods: tuple[str, ...]) -> dict:
    per_organism = {}
    for held, train_rows, test_rows in loo_folds(data, limit):
        test_y = data.labels[test_rows]
        absolute, contrastive, invariant = [], [], []
        for seed in seeds:
            if "absolute" in methods or "fused" in methods:
                absolute.append(refit_and_score(data, "absolute", train_rows,
                                                test_rows, max_epochs, seed))
            if "contrastive" in methods or "fused" in methods:
                contrastive.append(refit_and_score(data, "contrastive", train_rows,
                                                   test_rows, max_epochs, seed))
            if "invariant" in methods:
                invariant.append(refit_and_score(data, "absolute", train_rows,
                                                 test_rows, max_epochs, seed,
                                                 invariant=True))
        scores = {}
        if absolute:
            scores["absolute"] = np.mean(absolute, axis=0)
        if contrastive:
            scores["contrastive"] = np.mean(contrastive, axis=0)
        if absolute and contrastive:
            scores["fused"] = (0.5 * rank01(scores["absolute"])
                               + 0.5 * rank01(scores["contrastive"]))
        if invariant:
            scores["invariant"] = np.mean(invariant, axis=0)

        row = {"n": int(len(test_y))}
        row.update({method: auroc(test_y, scores[method])
                    for method in METHODS if method in scores and method in methods})
        per_organism[held] = row
        cells = "  ".join(f"{method[:4]}={row[method]:.4f}"
                          for method in METHODS if method in row)
        print(f"  {data.model}:{held:<26} n={row['n']:<5} {cells}", flush=True)
    return per_organism


def summarize(per_organism: dict, methods: tuple[str, ...]) -> dict:
    out = {}
    for method in (m for m in METHODS if m in methods):
        values = [row[method] for row in per_organism.values()
                  if method in row and not np.isnan(row[method])]
        out[method] = {
            "mean_auroc": float(np.mean(values)) if values else float("nan"),
            "worst_auroc": float(np.min(values)) if values else float("nan"),
            "worst_organism": (min((k for k in per_organism if method in per_organism[k]),
                                   key=lambda k: per_organism[k][method])
                               if values else None),
            "n_organisms": len(values),
        }
    return out


def print_summary(model: str, per_organism: dict, summary: dict):
    print(f"\n===== {model}: leave-one-organism-out ({len(per_organism)} folds) =====")
    print(f"{'method':<14}{'mean AUROC':>12}{'WORST AUROC':>14}   worst organism")
    for method, stats in summary.items():
        print(f"{method:<14}{stats['mean_auroc']:>12.4f}{stats['worst_auroc']:>14.4f}"
              f"   {stats['worst_organism']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-models", nargs="+", default=["qwen", "gemma"])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--limit-organisms", type=int, default=None,
                        help="smoke: use only the first N organisms")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS,
                        help="which probes to fit; 'fused' implies absolute+contrastive")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    print(f"device = {DEVICE}  seeds = {args.seeds}  max_epochs = {args.max_epochs}")
    manifest = load_manifest_rows()
    report = {}
    for model in args.base_models:
        data = AlignedData(model, manifest)
        organisms = sorted(set(data.organisms.tolist()))
        print(f"\n### {model}: {len(data.labels)} paired rows, {len(organisms)} organisms, "
              f"layer L{data.layer}, hidden {data.hidden_dim}")
        if data.skipped:
            print(f"    no contrastive cache (skipped): {len(data.skipped)} dataset(s)")
        if len(organisms) < 2:
            print("    fewer than 2 organisms -- LOO not defined, skipping")
            continue
        per_organism = run_model(data, args.max_epochs, args.seeds,
                                 args.limit_organisms, tuple(args.methods))
        if not per_organism:
            continue
        summary = summarize(per_organism, tuple(args.methods))
        report[model] = {"per_organism": per_organism, "summary": summary}
        print_summary(model, per_organism, summary)
        del data

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
