#!/usr/bin/env python3
"""Fit the sonic v3.1 bounded-refinement rule.

sonic v3 used a convex blend, ``(1-l)*probe_z + l*judge_z``.  The official
2026-07-26 result showed why that is unsafe: on Notus the probe ranks at about
`0.76` while the judge ranks at `0.864`, and at a probe weight of `0.65` the
probe overrides the judge and costs `0.1055` AUROC.  On Iris the same probe is
worth `+0.0427`.  A single global weight cannot serve both.

v3.1 makes the judge the ranking authority and lets the probe move a row by a
bounded amount only:

    score = judge_margin / JUDGE_SD
            + PROBE_CAP * tanh((probe_logit - PROBE_MEAN) / PROBE_SD)

The judge margin is bf16 and takes only about 34 distinct values per 400 rows,
so its ordering is full of ties.  ``PROBE_CAP`` is set to a small multiple of
one judge quantization step.  The probe therefore resolves ties and nudges
rows between neighbouring judge levels, but it can never overturn a confident
judge.  The loss is bounded even when the probe is far worse than random-guess
quality, which is exactly the Notus failure mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/nonlinear_probe"))
from token_probes import TokenProbe, build_token_probe_model

LAYER = 46
PROBE_DIR = {"gemma": "gemma_probe", "qwen": "qwen_probe", "nemotron": "nemotron_probe"}
HIDDEN_DIM = {"gemma": 5376, "qwen": 5120, "nemotron": 4096}
JUDGE_ULP = 0.125  # bf16 spacing of the label logits around magnitude 16-32
CAP_STEPS = 2.0    # probe may move a row by at most this many judge steps
PROBE_GAIN = 1.0


def load_probes() -> dict:
    """Load the three shipped probes exactly as the submission notebook does."""
    probes = {}
    for family in ("gemma", "qwen", "nemotron"):
        directory = REPO / "submission/whitebox_probe" / PROBE_DIR[family]
        config = json.load(open(directory / "config.json"))
        network = build_token_probe_model(config["architecture"], HIDDEN_DIM[family])
        network.load_state_dict(torch.load(directory / "model.pt", map_location="cpu"))
        network.eval()
        probe = TokenProbe(config["architecture"], seed=0, device="cpu",
                           batch_token_budget=8192)
        probe.model = network
        probe.feature_mean = torch.load(directory / "feature_mean.pt", map_location="cpu")
        probe.feature_std = torch.load(directory / "feature_std.pt", map_location="cpu")
        probes[family] = probe
    return probes


def family_of(dataset: str) -> str:
    """Return the base-model family a dev dataset belongs to."""
    if "Nemotron" in dataset:
        return "nemotron"
    return "gemma" if "gemma" in dataset else "qwen"


def shipped_probe_logits(probes: dict) -> dict:
    """Return ``dataset -> (index list, probe logits)`` from the shipped probes."""
    output = {}
    for path in sorted((REPO / "results/whitebox/activations").glob("*.tokens.npz")):
        dataset = path.name[: -len(".tokens.npz")].replace("__", "/")
        data = dict(np.load(path, allow_pickle=True))
        tokens_key = [key for key in data if key.startswith(f"tokens_L{LAYER}")][0]
        tokens = data[tokens_key]
        offsets = data["token_offsets"].astype(np.int64)
        flat = torch.from_numpy(tokens).float()
        logits = probes[family_of(dataset)].decision_function(flat, offsets)
        output[dataset] = ([str(value) for value in data["index"]],
                           np.nan_to_num(logits, nan=0.0))
    return output


def measure_scales(logits_by_dataset: dict) -> tuple[dict, dict]:
    """Return per-family ``(mean, sd)`` of the shipped probe's raw output.

    v3 divided by a spread measured from a different pass and never subtracted a
    mean.  Under a convex blend the mean is harmless, because a constant offset
    cannot change AUROC.  Under ``tanh`` it is not: an offset decides where each
    row sits on the curve, so it has to be measured and frozen too.
    """
    pooled: dict[str, list] = {}
    for dataset, (_, logits) in logits_by_dataset.items():
        pooled.setdefault(family_of(dataset), []).append(logits)
    means, sds = {}, {}
    for family, chunks in pooled.items():
        values = np.concatenate(chunks)
        means[family] = float(values.mean())
        sds[family] = float(values.std())
    return means, sds


def refine(judge_z: np.ndarray, probe_z: np.ndarray, cap: float) -> np.ndarray:
    """Judge ranking with a bounded probe correction."""
    return judge_z + cap * np.tanh(PROBE_GAIN * probe_z)


def degrade_to(probe_z: np.ndarray, labels: np.ndarray, target: float,
               rng: np.random.Generator) -> np.ndarray:
    """Add noise until the probe alone ranks at ``target`` AUROC.

    Notus is the only regime that matters and we have no Notus rows, so the
    check is a simulation: take a real dev fold and blunt the probe until it is
    as weak as the official Notus numbers say it is.
    """
    low, high = 0.0, 30.0
    for _ in range(40):
        middle = (low + high) / 2
        scored = np.mean([roc_auc_score(labels, probe_z + middle * rng.standard_normal(len(probe_z)))
                          for _ in range(7)])
        low, high = (middle, high) if scored > target else (low, middle)
    return probe_z + ((low + high) / 2) * rng.standard_normal(len(probe_z))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/ensemble_gate_eval/sonic_v3_1_constants.json")
    args = parser.parse_args(argv)

    previous = json.load(open(REPO / "results/ensemble_gate_eval/sonic_v3_constants.json"))
    judge_sd = previous["judge_margin_sd"]
    step = JUDGE_ULP / judge_sd
    cap = CAP_STEPS * step

    probes = load_probes()
    logits_by_dataset = shipped_probe_logits(probes)
    means, sds = measure_scales(logits_by_dataset)

    print("shipped-probe scale on dev (raw decision_function output)")
    for family in sorted(sds):
        old = previous["probe_logit_sd"][family]
        print(f"  {family:<9} mean {means[family]:+8.3f}  sd {sds[family]:7.3f}"
              f"   (v3 used sd {old:.3f}, off by x{sds[family] / old:.2f})")
    print(f"\njudge step = {JUDGE_ULP} / {judge_sd:.4f} = {step:.4f} z-units")
    print(f"PROBE_CAP  = {CAP_STEPS} steps = {cap:.4f}\n")

    folds = np.load(REPO / "results/ensemble_gate_eval/direct_blend_folds.npz",
                    allow_pickle=True)
    names = sorted({key.split("|")[0] for key in folds.keys()})

    print(f"{'regime':<16}{'judge':>9}{'v3 blend':>11}{'v3.1':>9}{'v3.1 worst':>12}")
    summary = {}
    for target in (0.70, 0.76, 0.85, None):
        rng = np.random.default_rng(7)
        base, blend_delta, refine_delta = [], [], []
        for name in names:
            family = name.split(":")[0]
            labels = folds[name + "|labels"]
            judge_z = folds[name + "|judge_margin"] / judge_sd
            probe_z = (folds[name + "|probe_logit"] - means[family]) / sds[family]
            if target is not None:
                probe_z = degrade_to(probe_z, labels, target, rng)
            judge_auroc = roc_auc_score(labels, judge_z)
            base.append(judge_auroc)
            blend_delta.append(roc_auc_score(labels, 0.65 * probe_z + 0.35 * judge_z)
                               - judge_auroc)
            refine_delta.append(roc_auc_score(labels, refine(judge_z, probe_z, cap))
                                - judge_auroc)
        label = f"probe @ {target}" if target else "probe as-is"
        summary[label] = {"judge": float(np.mean(base)),
                          "v3_blend_delta": float(np.mean(blend_delta)),
                          "v3_1_delta": float(np.mean(refine_delta)),
                          "v3_1_worst_fold_delta": float(min(refine_delta))}
        print(f"{label:<16}{np.mean(base):>9.4f}{np.mean(blend_delta):>+11.4f}"
              f"{np.mean(refine_delta):>+9.4f}{min(refine_delta):>+12.4f}")

    constants = {
        "rule": "judge_z + PROBE_CAP * tanh(PROBE_GAIN * (probe_logit - PROBE_MEAN) / PROBE_SD)",
        "judge_margin_sd": judge_sd,
        "judge_ulp": JUDGE_ULP,
        "judge_step_z": step,
        "probe_cap_steps": CAP_STEPS,
        "probe_cap": cap,
        "probe_gain": PROBE_GAIN,
        "probe_logit_mean": means,
        "probe_logit_sd": sds,
        "default_probe_mean": float(np.mean(list(means.values()))),
        "default_probe_sd": float(np.mean(list(sds.values()))),
        "threshold_judge_margin": 0.0,
        "threshold_probe_only": previous["threshold_probe_only"],
        "validation": summary,
    }
    args.output.write_text(json.dumps(constants, indent=2) + "\n")
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
