"""Score the sonic gate against the naive-ensemble baselines on leakage-free,
out-of-distribution folds.

For each base model, the probe is REFIT per fold (train rows only) and scored on
the held-out rows -- so the probe scores fed into the gate are leakage-free. The
judge is fixed (not refit); its per-row outputs are read from a cache produced
once by ``run_judge_local.py``. Every fusion method in ``gate.py`` is then scored
on the identical (probe, judge) inputs, per held-out organism, and aggregated.

Folds (probes are per-base-model, so folds stay within a base model):
  loo  -- leave-one-organism-out: train on all other organisms, test on the
          held-out one. Estimates "a new organism appears" (the OOD axis that
          matters; the frozen row-split cannot see it).
  xscn -- cross-scenario: train on one scenario, test on the other (base models
          with both scenarios only). The harder "a new deception style" axis.

LOCAL-DATA CAVEAT: the cached dev activations cover instructed/varied deception
across 21 organisms. They do NOT contain the private validation families
(Notus/Iris) -- so this reproduces the probe's cross-organism generalization
within the dev distribution, the best available local proxy, not Notus itself.

    python experiments/ensemble_gate_eval/run_eval.py                 # full
    python experiments/ensemble_gate_eval/run_eval.py --base-models qwen \
        --max-epochs 8 --limit-organisms 4   # fast smoke of the pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "experiments/nonlinear_probe"))

import gate  # noqa: E402
from token_probes import TokenProbe  # noqa: E402

ACTIVATIONS = REPO_ROOT / "results/whitebox/activations"
MANIFEST = REPO_ROOT / "dev_splits/manifest.csv"
SUBMISSION_PROBES = REPO_ROOT / "submission/whitebox_probe"
DEFAULT_JUDGE_CACHE = HERE / "judge_cache.json"
OUT_JSON = REPO_ROOT / "results/ensemble_gate_eval/report.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
PROBE_DIR = {"gemma": "gemma_probe", "qwen": "qwen_probe", "nemotron": "nemotron_probe"}
SCEN_SHORT = {"instructed-deception": "instr", "varied-deception": "varied"}


def base_model_of(dataset_name: str) -> str:
    if "gemma-3-27b-it" in dataset_name:
        return "gemma"
    if "Nemotron" in dataset_name:
        return "nemotron"
    if "Qwen3.5-27B" in dataset_name:
        return "qwen"
    raise ValueError(f"unknown base model for {dataset_name}")


def probe_layer(model: str) -> int:
    cfg = json.loads((SUBMISSION_PROBES / PROBE_DIR[model] / "config.json").read_text())
    return int(cfg["layer"])


def load_manifest_rows() -> list[dict]:
    with open(MANIFEST) as handle:
        return list(csv.DictReader(handle))


def cache_path(dataset_name: str) -> Path:
    return ACTIVATIONS / f"{dataset_name.replace('/', '__')}.tokens.npz"


class BaseModelData:
    """All dev rows for one base model, held as one flat token array plus
    per-row metadata, so any subset of rows can be repacked for a probe fit."""

    def __init__(self, model: str, manifest: list[dict]):
        self.model = model
        self.layer = probe_layer(model)
        datasets = sorted({r["dataset"] for r in manifest
                           if base_model_of(r["dataset"]) == model})
        spans, labels, organisms, keys = [], [], [], []
        for dataset in datasets:
            with np.load(cache_path(dataset), allow_pickle=True) as data:
                tokens = data[f"tokens_L{self.layer}"]
                offsets = data["token_offsets"].astype(np.int64)
                index = data["index"]
                deceptive = data["deceptive"].astype(int)
            meta = {(r["dataset"], str(r["index"])): r for r in manifest
                    if r["dataset"] == dataset}
            for row in range(len(index)):
                key = (dataset, str(index[row]))
                info = meta.get(key)
                if info is None:
                    continue
                spans.append(tokens[offsets[row]:offsets[row + 1]])
                labels.append(int(deceptive[row]))
                scen = SCEN_SHORT.get(info["scenario"], info["scenario"])
                lora = info["lora_id"] or "base"
                organisms.append(f"{scen}/{lora}")
                keys.append(key)
        lengths = [len(s) for s in spans]
        self.flat = np.concatenate(spans, axis=0)
        self.offsets = np.cumsum([0] + lengths).astype(np.int64)
        self.labels = np.asarray(labels)
        self.organisms = np.asarray(organisms, dtype=object)
        self.keys = keys
        self.scenario = np.asarray([o.split("/")[0] for o in organisms], dtype=object)

    def subset(self, rows: np.ndarray):
        """Repack the given global row ids into (flat tensor on DEVICE, offsets,
        labels, keys) for a probe fit or scoring pass."""
        rows = np.asarray(rows)
        pieces = [self.flat[self.offsets[r]:self.offsets[r + 1]] for r in rows]
        lengths = [len(p) for p in pieces]
        flat = np.concatenate(pieces, axis=0)
        offsets = np.cumsum([0] + lengths).astype(np.int64)
        finfo = torch.finfo(torch.float16)
        flat_t = torch.from_numpy(flat).clamp(finfo.min, finfo.max).to(DEVICE)
        return flat_t, offsets, self.labels[rows], [self.keys[r] for r in rows]


def refit_and_score(data: BaseModelData, train_rows, test_rows, max_epochs):
    """Refit the probe on train_rows, score test_rows. Leakage-free: the probe
    only ever sees train rows (and its own internal validation split of them)."""
    train_flat, train_off, train_y, _ = data.subset(train_rows)
    probe = TokenProbe("transformer", seed=SEED, device=DEVICE,
                       max_epochs=max_epochs).fit(train_flat, train_off, train_y)
    test_flat, test_off, test_y, test_keys = data.subset(test_rows)
    scores = probe.predict_proba(test_flat, test_off)[:, 1]
    return scores, test_y, test_keys


def loo_folds(data: BaseModelData, limit: int | None):
    organisms = sorted(set(data.organisms.tolist()))
    if len(organisms) < 2:
        return
    if limit:
        organisms = organisms[:limit]
    all_rows = np.arange(len(data.labels))
    for held in organisms:
        test_rows = all_rows[data.organisms == held]
        train_rows = all_rows[data.organisms != held]
        if limit:  # smoke: restrict training pool to the kept organisms
            keep = np.isin(data.organisms, organisms)
            train_rows = all_rows[keep & (data.organisms != held)]
        if len(np.unique(data.labels[train_rows])) < 2:
            continue
        yield held, train_rows, test_rows


def xscn_folds(data: BaseModelData):
    scenarios = sorted(set(data.scenario.tolist()))
    if len(scenarios) < 2:
        return
    all_rows = np.arange(len(data.labels))
    for train_scen in scenarios:
        for test_scen in scenarios:
            if train_scen == test_scen:
                continue
            train_rows = all_rows[data.scenario == train_scen]
            test_rows = all_rows[data.scenario == test_scen]
            yield f"{train_scen}->{test_scen}", train_rows, test_rows


def load_judge_cache(path: Path):
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {tuple(k.split("|", 1)): (float(v[0]), float(v[1])) for k, v in raw.items()}


def score_group(probe, judge_hard, judge_soft, labels):
    """Run every fusion method on one held-out group and return per-method
    AUROC + balanced accuracy."""
    out = {}
    both_classes = len(np.unique(labels)) > 1
    for name, fn in gate.METHODS.items():
        needs_judge = name != "probe_only"
        if needs_judge and judge_hard is None:
            continue
        res = fn(probe, judge_hard, judge_soft)
        auroc = roc_auc_score(labels, res.score) if both_classes else float("nan")
        ba = balanced_accuracy_score(labels, res.deceptive)
        out[name] = {"auroc": float(auroc), "balanced_accuracy": float(ba)}
    return out


def aggregate(per_organism: dict) -> dict:
    methods = defaultdict(lambda: {"auroc": [], "balanced_accuracy": []})
    for metrics in per_organism.values():
        for method, m in metrics.items():
            if not np.isnan(m["auroc"]):
                methods[method]["auroc"].append(m["auroc"])
            methods[method]["balanced_accuracy"].append(m["balanced_accuracy"])
    return {method: {"mean_auroc": float(np.mean(v["auroc"])) if v["auroc"] else float("nan"),
                     "mean_ba": float(np.mean(v["balanced_accuracy"])),
                     "n_organisms": len(v["balanced_accuracy"])}
            for method, v in methods.items()}


def print_table(title: str, per_organism: dict, agg: dict):
    methods = list(gate.METHODS.keys())
    present = [m for m in methods if m in agg]
    print(f"\n===== {title} =====")
    head = f"{'organism':<22}" + "".join(f"{m[:14]:>15}" for m in present)
    print(head + "   (BA / AUROC)")
    for organism in sorted(per_organism):
        row = f"{organism:<22}"
        for m in present:
            cell = per_organism[organism].get(m)
            if cell is None:
                row += f"{'-':>15}"
            else:
                a = cell["auroc"]
                row += f"{cell['balanced_accuracy']:.3f}/{'' if np.isnan(a) else f'{a:.3f}':>5}".rjust(15)
        print(row)
    print("-" * len(head))
    mean_row = f"{'MEAN':<22}"
    for m in present:
        mean_row += f"{agg[m]['mean_ba']:.3f}/{agg[m]['mean_auroc']:.3f}".rjust(15)
    print(mean_row)


def run_foldset(name, folds, data, judge_cache, max_epochs):
    per_organism = {}
    for held, train_rows, test_rows in folds:
        scores, test_y, test_keys = refit_and_score(data, train_rows, test_rows, max_epochs)
        if judge_cache is not None:
            jh, js, keep = [], [], []
            for i, key in enumerate(test_keys):
                hit = judge_cache.get(key)
                if hit is not None:
                    jh.append(hit[0]); js.append(hit[1]); keep.append(i)
            if len(keep) == len(test_keys):
                judge_hard = np.asarray(jh); judge_soft = np.asarray(js)
            else:
                judge_hard = judge_soft = None  # partial coverage -> probe-only
        else:
            judge_hard = judge_soft = None
        per_organism[f"{data.model}:{held}"] = score_group(scores, judge_hard, judge_soft, test_y)
        print(f"  [{name}] {data.model}:{held:<24} "
              f"n={len(test_y):<4} scored "
              f"{'probe+judge' if judge_hard is not None else 'probe-only'}", flush=True)
    return per_organism


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-models", nargs="+", default=["qwen", "gemma", "nemotron"])
    parser.add_argument("--folds", nargs="+", default=["loo", "xscn"],
                        choices=["loo", "xscn"])
    parser.add_argument("--judge-cache", type=Path, default=DEFAULT_JUDGE_CACHE)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--limit-organisms", type=int, default=None,
                        help="LOO smoke: use only the first N organisms")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    judge_cache = load_judge_cache(args.judge_cache)
    print(f"device = {DEVICE}")
    print(f"judge cache = {args.judge_cache if judge_cache else 'MISSING -> probe-only baselines'} "
          f"({len(judge_cache) if judge_cache else 0} rows)")

    manifest = load_manifest_rows()
    report = {}
    for model in args.base_models:
        data = BaseModelData(model, manifest)
        n_org = len(set(data.organisms.tolist()))
        print(f"\n### {model}: {len(data.labels)} rows, {n_org} organisms, layer L{data.layer}")
        report[model] = {}
        if "loo" in args.folds:
            per = run_foldset("loo", loo_folds(data, args.limit_organisms), data,
                              judge_cache, args.max_epochs)
            if per:
                agg = aggregate(per)
                report[model]["loo"] = {"per_organism": per, "aggregate": agg}
                print_table(f"{model} — leave-one-organism-out", per, agg)
        if "xscn" in args.folds:
            per = run_foldset("xscn", xscn_folds(data), data, judge_cache, args.max_epochs)
            if per:
                agg = aggregate(per)
                report[model]["xscn"] = {"per_organism": per, "aggregate": agg}
                print_table(f"{model} — cross-scenario", per, agg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
