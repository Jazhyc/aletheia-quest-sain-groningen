"""Train the shipping shared-trunk probe and export it as per-family weights.

The evaluation harness (`run_multifamily_eval.py`) only ever fits leave-one-out
folds. This script fits the trunk on EVERY dev row, which is the weight set that
ships, then splits it into the one-probe-per-family layout the notebook already
loads.

The split is exact. `MultiFamilyTokenProbe` is a `TransformerTokenProbe` whose
`projection` has been moved into a `ModuleDict`; the encoder, the head and the
positional encoding are shared and unchanged. So for one family we copy the
shared encoder and head, plus that family's projection, into a plain
`TransformerTokenProbe`. Its `state_dict` keys then match what cell 5 of the
notebook builds, and no notebook code has to change.

The export is verified, not assumed: every family's rows are scored with the
shared model and with the exported single-family model, and the run fails unless
the two agree to 1e-4.

nemotron and qwen are exported. gemma is NOT. A 3-seed sweep gives, per family,
the shared trunk minus the single-family probe on mean AUROC:

    nemotron  +0.0728 / +0.0159 / +0.0253   (row CV; positive on every seed)
    qwen      +0.0033 / +0.0006 / +0.0002   (positive on every seed, but tiny)
    gemma     -0.0168 / -0.0158 / -0.0107   (negative on every seed)

qwen ships on the shared trunk despite the small margin. The offline folds
cannot reach the Notus regime -- every local fold is easier than every Notus
unit -- so "no measurable gain here" is not "no gain there". The measured
downside is near zero, Notus/Qwen is 42.5% of the leaderboard gap, and the
trunk's demonstrated property is variance reduction, which is what transfer
needs. gemma keeps its own probe: the loss repeats on all three seeds,
Notus/gemma is already our best Notus unit, and it is only 11% of the gap.
See `experiments/nonlinear_probe/MULTIFAMILY_EVALUATION.md`.

    python experiments/nonlinear_probe/export_multifamily_probe.py --dry-run
    python experiments/nonlinear_probe/export_multifamily_probe.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments/ensemble_gate_eval"))
sys.path.insert(0, str(HERE))

from multifamily_probe import MultiFamilyProbe  # noqa: E402
from run_eval import BaseModelData, load_manifest_rows  # noqa: E402
from token_probes import TokenProbe, TransformerTokenProbe  # noqa: E402

PROBE_ROOT = REPO_ROOT / "submission/whitebox_probe"
PROBE_DIR = {"gemma": "gemma_probe", "qwen": "qwen_probe", "nemotron": "nemotron_probe"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAMILIES = ("qwen", "gemma", "nemotron")
EXPORT_FAMILIES = ("nemotron", "qwen")
PARITY_TOLERANCE = 1e-4


def subset(data: BaseModelData, rows: np.ndarray):
    """Repack rows of one family into (flat tensor on DEVICE, offsets)."""
    pieces = [data.flat[data.offsets[r]:data.offsets[r + 1]] for r in np.asarray(rows)]
    packed = np.concatenate(pieces, axis=0)
    offsets = np.cumsum([0] + [len(p) for p in pieces]).astype(np.int64)
    finfo = torch.finfo(torch.float16)
    return torch.from_numpy(packed).clamp(finfo.min, finfo.max).to(DEVICE), offsets


def export_family(shared: MultiFamilyProbe, family: str,
                  hidden_dim: int) -> TransformerTokenProbe:
    """
    Copy the shared trunk plus one family's projection into a plain probe.

    :param shared: The fitted shared-trunk probe.
    :param family: Which family's projection to take.
    :param hidden_dim: That family's residual-stream width.
    :return: A `TransformerTokenProbe` in eval mode with the merged weights.
    """
    single = TransformerTokenProbe(hidden_dim)
    merged = {}
    for key, value in shared.model.state_dict().items():
        if key.startswith("projections."):
            prefix = f"projections.{family}."
            if key.startswith(prefix):
                merged[key.replace(prefix, "projection.")] = value
        else:
            merged[key] = value
    single.load_state_dict(merged)
    return single.to(DEVICE).eval()


def parity_check(shared: MultiFamilyProbe, single: TransformerTokenProbe,
                 family: str, data: BaseModelData) -> float:
    """
    Score every row of one family two ways and return the largest difference.

    The single-family path is driven through `TokenProbe`, which is the same
    standardize-pack-forward code the notebook runs, so this checks the export
    end to end rather than just comparing tensors.
    """
    flat, offsets = subset(data, np.arange(len(data.labels)))
    shared_scores = shared.predict_proba(family, flat, offsets)[:, 1]

    wrapper = TokenProbe("transformer", device=DEVICE)
    wrapper.model = single
    wrapper.feature_mean, wrapper.feature_std = shared.moments[family]
    single_scores = wrapper.predict_proba(flat, offsets)[:, 1]

    del flat
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return float(np.abs(shared_scores - single_scores).max())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="fit, export and verify, but write nothing")
    parser.add_argument("--export-families", nargs="*", default=list(EXPORT_FAMILIES),
                        help="families to export. gemma is excluded by default: a "
                             "3-seed sweep and a leave-one-cell-out rerun both show "
                             "the trunk losing to its single-family probe. Pass all "
                             "three only to build a deliberate trunk diagnostic.")
    parser.add_argument("--balanced", action="store_true",
                        help="fit with the balanced recipe (grouped early-stopping "
                             "split + base rows upweighted to ~50% of the loss) "
                             "instead of the shipped random row split")
    parser.add_argument("--out-root", type=Path, default=PROBE_ROOT,
                        help="destination probe root; point elsewhere to avoid "
                             "overwriting the shipped weights")
    args = parser.parse_args(argv)

    print(f"device = {DEVICE}  seed = {args.seed}  max_epochs = {args.max_epochs}")
    manifest = load_manifest_rows()
    families, family_data = {}, {}
    for family in FAMILIES:
        data = BaseModelData(family, manifest)
        families[family] = data
        flat, offsets = subset(data, np.arange(len(data.labels)))
        family_data[family] = (flat, offsets, data.labels)
        print(f"  {family}: {len(data.labels)} rows, {int(data.labels.sum())} positive, "
              f"hidden {data.flat.shape[1]}", flush=True)

    print("\nfitting the shared trunk on every dev row ...", flush=True)
    started = time.time()
    fit_kwargs = {}
    if args.balanced:
        from train_v3_3_probe import base_balanced_weights
        fit_kwargs = {
            "family_groups": {f: families[f].organisms for f in FAMILIES},
            "family_weights": {f: base_balanced_weights(families[f].organisms)
                               for f in FAMILIES},
        }
        print("  recipe: balanced (grouped split + base upweighting)", flush=True)
    else:
        print("  recipe: shipped (random row split, uniform weights)", flush=True)
    shared = MultiFamilyProbe(seed=args.seed, device=DEVICE,
                              max_epochs=args.max_epochs).fit(family_data, **fit_kwargs)
    print(f"  done in {time.time() - started:.0f}s", flush=True)

    exported = {}
    for family in args.export_families:
        hidden_dim = int(families[family].flat.shape[1])
        single = export_family(shared, family, hidden_dim)
        difference = parity_check(shared, single, family, families[family])
        status = "OK" if difference <= PARITY_TOLERANCE else "FAILED"
        print(f"  parity {family}: max |shared - exported| = {difference:.2e}  {status}")
        if difference > PARITY_TOLERANCE:
            raise SystemExit(f"export parity failed for {family}: {difference:.2e}")
        exported[family] = (single, hidden_dim)

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    stamp = time.strftime("%Y%m%dT%H%M%S")
    for family, (single, hidden_dim) in exported.items():
        target = args.out_root / PROBE_DIR[family]
        target.mkdir(parents=True, exist_ok=True)
        backup = args.out_root / f"{PROBE_DIR[family]}.backup.{stamp}"
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "feature_mean.pt", "feature_std.pt", "config.json"):
            if (target / name).exists():
                shutil.copy2(target / name, backup / name)

        mean, std = shared.moments[family]
        torch.save(single.state_dict(), target / "model.pt")
        torch.save(mean.detach().cpu(), target / "feature_mean.pt")
        torch.save(std.detach().cpu(), target / "feature_std.pt")

        # A fresh --out-root has no config yet; seed it from the shipped probe
        # so layer/architecture/batch settings stay identical.
        template = target / "config.json"
        if not template.exists():
            template = PROBE_ROOT / PROBE_DIR[family] / "config.json"
        config = json.loads(template.read_text())
        config["hidden_dim"] = hidden_dim
        config["trained_by"] = "shared_trunk_multifamily"
        config["shared_trunk_families"] = list(FAMILIES)
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"  wrote {target}  (backup in {backup.name})")

    skipped = [f for f in FAMILIES if f not in args.export_families]
    if skipped:
        print(f"\nnot exported (keeping existing weights): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
