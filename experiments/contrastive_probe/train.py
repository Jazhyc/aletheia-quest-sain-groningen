"""
Train contrastive activation probes on difference features.

Loads .npz files from contrastive extraction, trains one probe per base model
family (qwen/gemma/nemotron) on the per-token activation differences, and
evaluates on standardized 80/10/10 splits.

Outputs probe weights to submission/whitebox_probe/{family}_probe/contrastive/
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "nonlinear_probe"))
from token_probes import TokenProbe  # noqa: E402


def load_contrastive_data(activations_dir: Path, family: str) -> tuple:
    """Load all .npz files for a base model family. Returns (flat_diff, offsets, labels)."""
    import re
    pattern = re.compile(rf".*{family}.*\.npz", re.IGNORECASE) if family != "nemotron" else re.compile(r".*nemotron.*\.npz", re.IGNORECASE)
    files = sorted(p for p in activations_dir.glob("*.npz") if pattern.search(p.name))
    if not files:
        raise FileNotFoundError(f"No .npz files found for family={family} in {activations_dir}")

    diff_parts, offset_parts, label_parts = [], [], []
    offset_cursor = 0

    for fp in files:
        data = np.load(fp, allow_pickle=True)
        d = data["diff_flat"]
        o = data["offsets"]
        l = data["deceptive"]

        if d.shape[0] == 0:
            continue

        diff_parts.append(torch.from_numpy(d.astype(np.float32)))
        # offsets are row boundaries; adjust for concatenation
        adjusted = o[1:].copy() + offset_cursor
        full_offsets = np.concatenate([[0], adjusted])
        offset_parts.append(full_offsets)
        offset_cursor = full_offsets[-1]
        label_parts.append(l)

    if not diff_parts:
        raise ValueError(f"No valid data for family={family}")

    flat_diff = torch.cat(diff_parts)
    all_offsets = np.concatenate(offset_parts) if len(offset_parts) > 1 else offset_parts[0]
    all_labels = np.concatenate(label_parts) if len(label_parts) > 1 else label_parts[0]

    return flat_diff, all_offsets, all_labels


def train_family_probe(flat_diff, offsets, labels, family: str,
                       out_dir: Path, seed: int = 0, device: str = "cuda"):
    """Train a transformer token probe and save weights."""
    print(f"\n{'='*60}")
    print(f"Training contrastive probe: {family}")
    print(f"  {flat_diff.shape[0]} tokens, {len(offsets) - 1} examples")
    print(f"  deceptive rate: {labels.mean():.3f}")

    # Train on all data (same protocol as shipped probes — fit on everything
    # since dev data is what we're allowed to use)
    probe = TokenProbe("transformer", seed=seed, device=device, max_epochs=60)
    probe.fit(flat_diff, offsets, labels)

    # Save
    save_dir = out_dir / f"{family}_probe" / "contrastive"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(probe.model.state_dict(), save_dir / "model.pt")
    torch.save(probe.feature_mean, save_dir / "feature_mean.pt")
    torch.save(probe.feature_std, save_dir / "feature_std.pt")

    config = {
        "architecture": "transformer",
        "layer": 46,
        "hidden_dim": flat_diff.shape[1],
        "d_model": 128,
        "n_heads": 4,
        "dim_feedforward": 256,
        "n_blocks": 2,
        "dropout": 0.1,
        "extract_token_budget": 2560,
        "extract_max_batch": 32,
        "batch_token_budget": 8192,
    }
    json.dump(config, open(save_dir / "config.json", "w"), indent=2)

    print(f"  Saved to {save_dir}")
    return probe


def evaluate_probe(flat_diff, offsets, labels, probe, device="cuda"):
    """Quick evaluation using the probe's own score method."""
    probs = probe.predict_proba(flat_diff, offsets)
    # predict_proba returns (N, 2) — take P(deceptive) = column 1
    scores = probs[:, 1] if probs.ndim == 2 else probs
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    auroc = roc_auc_score(labels, scores)
    preds = (scores >= 0.5).astype(int)
    ba = balanced_accuracy_score(labels, preds)

    return {"auroc": auroc, "balanced_accuracy": ba}


def main():
    parser = argparse.ArgumentParser(description="Train contrastive probes")
    parser.add_argument("--activations-dir", type=Path,
                        default=REPO_ROOT / "results" / "contrastive" / "activations")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "submission" / "whitebox_probe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--family", choices=["qwen", "gemma", "nemotron", "all"],
                        default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    families = ["qwen", "gemma", "nemotron"] if args.family == "all" else [args.family]

    results = {}
    for family in families:
        try:
            flat_diff, offsets, labels = load_contrastive_data(args.activations_dir, family)
        except (FileNotFoundError, ValueError) as e:
            print(f"SKIP {family}: {e}")
            continue

        probe = train_family_probe(flat_diff, offsets, labels, family,
                                   args.out_dir, seed=args.seed, device=args.device)

        metrics = evaluate_probe(flat_diff, offsets, labels, probe, device=args.device)
        results[family] = metrics
        print(f"  In-sample eval: AUROC={metrics['auroc']:.4f}, BA={metrics['balanced_accuracy']:.4f}")
        print(f"  (In-sample = trained on all dev data; real transfer tested via dry run)")

    if results:
        print(f"\n{'='*60}")
        print("Summary:")
        for fam, m in results.items():
            print(f"  {fam}: AUROC={m['auroc']:.4f}, BA={m['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
