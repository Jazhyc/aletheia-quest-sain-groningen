#!/usr/bin/env python3
"""Correlate probe AUROC / BA with prompt length.

Uses the v4.3 probe logits cache (pre-computed L46/L40 probe logits for all dev
datasets).  For each dataset the full HuggingFace dataset is loaded to compute
prompt lengths via the per-family tokenizer.

Two levels of analysis:

1. **Per-dataset**: average prompt length (full & response-only) vs probe AUROC / BA.
   One point per dataset → Spearman/Pearson r across datasets.

2. **Per-bin**: bin examples by prompt-length decile (pooled across datasets or
   per family), compute AUROC / BA per bin.  This answers: "does the probe
   degrade on longer prompts within the same distribution?"

Run:
    python experiments/ensemble_gate_eval/correlate_length_probe_perf.py

If the HF datasets aren't locally available, pass ``--no-datasets`` to use only the
response token counts from the activation files (cheaper, no network/tokenizer needed).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "results/ensemble_gate_eval/v4_3_probe_logits_cache.npz"
ACTIVATIONS_DIR = REPO_ROOT / "results/whitebox/activations"
MANIFEST = REPO_ROOT / "dev_splits/manifest.csv"

# v4.3 frozen standardisation constants (L40 from sonic_v4_L40_constants.json, L46 from v3_3_mini)
FROZEN_46 = json.load(open(REPO_ROOT / "results/ensemble_gate_eval/sonic_v3_3_mini_constants.json"))
FROZEN_40 = json.load(open(REPO_ROOT / "results/ensemble_gate_eval/sonic_v4_L40_constants.json"))


def family_of(dataset: str) -> str:
    if "Nemotron" in dataset:
        return "nemotron"
    return "gemma" if "gemma" in dataset.lower() else "qwen"


MODEL_TO_FAMILY_TOKENIZER = {}
MODEL_IDS = {
    "gemma": "google/gemma-3-27b-it",
    "qwen": "Qwen/Qwen3.5-27B",
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}


def get_tokenizer(family: str):
    """Lazily load tokenizer per family to compute prompt lengths."""
    if family not in MODEL_TO_FAMILY_TOKENIZER:
        from transformers import AutoTokenizer
        MODEL_TO_FAMILY_TOKENIZER[family] = AutoTokenizer.from_pretrained(MODEL_IDS[family])
    return MODEL_TO_FAMILY_TOKENIZER[family]


def tokenize_messages(messages: list, tokenizer) -> int:
    """Return the full token count of the conversation (chat template).

    ``apply_chat_template(tokenize=True)`` returns a BatchEncoding when
    ``return_dict`` is omitted (its default), whose ``len()`` is the number of
    dict keys (2), not the token count.  ``return_dict=True`` returns the same
    BatchEncoding, so we access ``["input_ids"]`` explicitly.
    """
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True)
        return len(encoded["input_ids"])
    except Exception:
        text = "\n".join(m.get("content", "") for m in messages)
        return len(tokenizer.encode(text, add_special_tokens=True))


def response_length_messages(messages: list, tokenizer) -> int:
    """Token count of the final assistant response only."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return len(tokenizer.encode(m.get("content", ""), add_special_tokens=False))
    return 0


def load_cache() -> dict[str, dict]:
    """Load v4.3 probe logits cache, indexed by dataset name."""
    blobs = dict(np.load(CACHE, allow_pickle=True))
    datasets = set()
    for key in blobs:
        if "|" in key:
            datasets.add(key.split("|")[0])
    result = {}
    for ds in sorted(datasets):
        result[ds] = {
            "L46": blobs.get(f"{ds}|L46"),
            "L40": blobs.get(f"{ds}|L40"),
            "labels": blobs.get(f"{ds}|labels"),
            "margin": blobs.get(f"{ds}|margin"),
        }
    return result


def probe_z_scores(logits_l46, logits_l40, family: str):
    """Standardise both probes with frozen per-family moments, then fuse."""
    mean_46 = FROZEN_46["probe_logit_mean"][family]
    sd_46 = FROZEN_46["probe_logit_sd"][family]
    mean_40 = FROZEN_40["probe_logit_mean"][family]
    sd_40 = FROZEN_40["probe_logit_sd"][family]
    z46 = (logits_l46 - mean_46) / max(sd_46, 1e-8)
    z40 = (logits_l40 - mean_40) / max(sd_40, 1e-8)
    return (z46 + z40) / 2.0


def load_prompt_lengths_from_datasets(datasets: set[str]) -> dict[str, np.ndarray]:
    """For each dataset, load HF dataset and compute per-example prompt lengths.

    Returns dict[dataset] -> array of (full_tokens, response_tokens, char_count) per example,
    aligned with the cache ordering (by index).
    """
    from datasets import load_dataset

    result = {}
    for ds_name in sorted(datasets):
        try:
            ds = load_dataset(ds_name, split="test")
        except Exception as e:
            print(f"  SKIP {ds_name.split('/')[-1]}: {e}", flush=True)
            continue

        family = family_of(ds_name)
        tok = get_tokenizer(family)

        indices = list(ds["index"])
        messages_list = ds["messages"]

        full_tokens = np.array([tokenize_messages(m, tok) for m in messages_list])
        resp_tokens = np.array([response_length_messages(m, tok) for m in messages_list])
        chars = np.array([sum(len(msg.get("content", "")) for msg in msgs)
                          for msgs in messages_list])

        # Build index → position map so we can align with cache ordering
        idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}

        result[ds_name] = {
            "full_tokens": full_tokens,
            "resp_tokens": resp_tokens,
            "chars": chars,
            "index_to_pos": idx_to_pos,
            "indices": indices,
        }
        short = ds_name.split("/")[-1]
        print(f"  {short}: {len(indices)} ex, full={full_tokens.mean():.0f}±{full_tokens.std():.0f} tok, "
              f"resp={resp_tokens.mean():.0f}±{resp_tokens.std():.0f} tok", flush=True)
    return result


def load_prompt_lengths_from_activations(datasets: set[str]) -> dict[str, np.ndarray]:
    """Fallback: use response_token counts from the .tokens.npz activation files.

    These only have response token counts (not full prompt), but they're
    available without HF tokenizers/datasets.
    """
    result = {}
    for ds_name in sorted(datasets):
        safe = ds_name.replace("/", "__")
        token_file = ACTIVATIONS_DIR / f"{safe}.tokens.npz"
        if not token_file.exists():
            print(f"  SKIP {ds_name.split('/')[-1]}: no tokens file", flush=True)
            continue
        data = dict(np.load(token_file, allow_pickle=True))
        indices = list(data["index"])
        resp_tokens = data["response_tokens"].astype(np.int32)

        idx_to_pos = {int(idx): pos for pos, idx in enumerate(indices)}

        result[ds_name] = {
            "resp_tokens": resp_tokens,
            "indices": indices,
            "index_to_pos": idx_to_pos,
        }
        short = ds_name.split("/")[-1]
        print(f"  {short}: {len(indices)} ex, resp={resp_tokens.mean():.0f}±{resp_tokens.std():.0f} tok", flush=True)
    return result


def compute_per_dataset_metrics(cache, lengths, use_full_tokens: bool = True):
    """Compute per-dataset probe AUROC/BA alongside average prompt length.

    Returns list of (dataset_short, family, avg_length, std_length,
    probe_auroc_fused, probe_auroc_L46, probe_auroc_L40, probe_ba_fused, n_examples).
    """
    rows = []
    for ds_name, entry in cache.items():
        if ds_name not in lengths:
            continue
        len_data = lengths[ds_name]

        labels = entry["labels"]
        l46 = entry["L46"]
        l40 = entry["L40"]
        if labels is None or l46 is None or l40 is None:
            continue
        n = len(labels)
        if n < 2 or len(np.unique(labels)) < 2:
            continue

        if use_full_tokens and "full_tokens" in len_data:
            length_arr = len_data["full_tokens"]
        else:
            length_arr = len_data["resp_tokens"]

        family = family_of(ds_name)

        # Probe z-scores
        z_fused = probe_z_scores(l46, l40, family)
        z_46 = (l46 - FROZEN_46["probe_logit_mean"][family]) / max(FROZEN_46["probe_logit_sd"][family], 1e-8)
        z_40 = (l40 - FROZEN_40["probe_logit_mean"][family]) / max(FROZEN_40["probe_logit_sd"][family], 1e-8)

        # Scores from z
        fused_score = 1.0 / (1.0 + np.exp(-np.clip(z_fused, -80, 80)))
        score_46 = 1.0 / (1.0 + np.exp(-np.clip(z_46, -80, 80)))
        score_40 = 1.0 / (1.0 + np.exp(-np.clip(z_40, -80, 80)))

        auroc_fused = roc_auc_score(labels, fused_score)
        auroc_46 = roc_auc_score(labels, score_46)
        auroc_40 = roc_auc_score(labels, score_40)
        ba_fused = balanced_accuracy_score(labels, fused_score >= 0.5)

        short = ds_name.split("/")[-1]
        rows.append((short, family, float(length_arr.mean()), float(length_arr.std()),
                     auroc_fused, auroc_46, auroc_40, ba_fused, n))
    return rows


def compute_per_bin_metrics(cache, lengths, n_bins: int = 10, use_full_tokens: bool = True):
    """Bin examples by prompt length (pooled across datasets) and compute AUROC/BA per bin."""
    all_lengths = []
    all_z_fused = []
    all_z_46 = []
    all_z_40 = []
    all_labels = []

    for ds_name, entry in cache.items():
        if ds_name not in lengths:
            continue
        len_data = lengths[ds_name]
        labels = entry["labels"]
        l46 = entry["L46"]
        l40 = entry["L40"]
        if labels is None or l46 is None or l40 is None:
            continue
        if len(np.unique(labels)) < 2:
            continue

        if use_full_tokens and "full_tokens" in len_data:
            length_arr = len_data["full_tokens"]
        else:
            length_arr = len_data["resp_tokens"]

        family = family_of(ds_name)
        z_fused = probe_z_scores(l46, l40, family)
        z_46 = (l46 - FROZEN_46["probe_logit_mean"][family]) / max(FROZEN_46["probe_logit_sd"][family], 1e-8)
        z_40 = (l40 - FROZEN_40["probe_logit_mean"][family]) / max(FROZEN_40["probe_logit_sd"][family], 1e-8)

        for i in range(len(labels)):
            all_lengths.append(length_arr[i])
            all_z_fused.append(z_fused[i])
            all_z_46.append(z_46[i])
            all_z_40.append(z_40[i])
            all_labels.append(labels[i])

    all_lengths = np.array(all_lengths)
    all_z_fused = np.array(all_z_fused)
    all_z_46 = np.array(all_z_46)
    all_z_40 = np.array(all_z_40)
    all_labels = np.array(all_labels)

    lo, hi = all_lengths.min(), all_lengths.max()
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    rows = []
    for b in range(n_bins):
        mask = (all_lengths >= bin_edges[b]) & (all_lengths < bin_edges[b + 1])
        if b == n_bins - 1:
            mask = (all_lengths >= bin_edges[b]) & (all_lengths <= bin_edges[b + 1])
        if mask.sum() < 2 or len(np.unique(all_labels[mask])) < 2:
            continue
        fused_score = 1.0 / (1.0 + np.exp(-np.clip(all_z_fused[mask], -80, 80)))
        score_46 = 1.0 / (1.0 + np.exp(-np.clip(all_z_46[mask], -80, 80)))
        score_40 = 1.0 / (1.0 + np.exp(-np.clip(all_z_40[mask], -80, 80)))
        rows.append((
            float(bin_edges[b]), float(bin_edges[b + 1]),
            int(mask.sum()),
            roc_auc_score(all_labels[mask], fused_score),
            roc_auc_score(all_labels[mask], score_46),
            roc_auc_score(all_labels[mask], score_40),
            balanced_accuracy_score(all_labels[mask], fused_score >= 0.5),
        ))
    return rows


def compute_per_family_bin_metrics(cache, lengths, n_bins: int = 8, use_full_tokens: bool = True):
    """Bin examples by prompt length WITHIN each model family."""
    per_family = defaultdict(lambda: {"lengths": [], "z_fused": [], "z_46": [], "z_40": [], "labels": []})

    for ds_name, entry in cache.items():
        if ds_name not in lengths:
            continue
        len_data = lengths[ds_name]
        labels = entry["labels"]
        l46 = entry["L46"]
        l40 = entry["L40"]
        if labels is None or l46 is None or l40 is None:
            continue

        if use_full_tokens and "full_tokens" in len_data:
            length_arr = len_data["full_tokens"]
        else:
            length_arr = len_data["resp_tokens"]

        family = family_of(ds_name)
        z_fused = probe_z_scores(l46, l40, family)
        z_46 = (l46 - FROZEN_46["probe_logit_mean"][family]) / max(FROZEN_46["probe_logit_sd"][family], 1e-8)
        z_40 = (l40 - FROZEN_40["probe_logit_mean"][family]) / max(FROZEN_40["probe_logit_sd"][family], 1e-8)

        for i in range(len(labels)):
            per_family[family]["lengths"].append(length_arr[i])
            per_family[family]["z_fused"].append(z_fused[i])
            per_family[family]["z_46"].append(z_46[i])
            per_family[family]["z_40"].append(z_40[i])
            per_family[family]["labels"].append(labels[i])

    result = {}
    for family, data in sorted(per_family.items()):
        lens = np.array(data["lengths"])
        z_f = np.array(data["z_fused"])
        z_46 = np.array(data["z_46"])
        z_40 = np.array(data["z_40"])
        labs = np.array(data["labels"])

        lo, hi = lens.min(), lens.max()
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        rows = []
        for b in range(n_bins):
            mask = (lens >= bin_edges[b]) & (lens < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (lens >= bin_edges[b]) & (lens <= bin_edges[b + 1])
            if mask.sum() < 5 or len(np.unique(labs[mask])) < 2:
                continue
            fused_score = 1.0 / (1.0 + np.exp(-np.clip(z_f[mask], -80, 80)))
            score_46 = 1.0 / (1.0 + np.exp(-np.clip(z_46[mask], -80, 80)))
            score_40 = 1.0 / (1.0 + np.exp(-np.clip(z_40[mask], -80, 80)))
            rows.append((
                float(bin_edges[b]), float(bin_edges[b + 1]),
                int(mask.sum()),
                roc_auc_score(labs[mask], fused_score),
                roc_auc_score(labs[mask], score_46),
                roc_auc_score(labs[mask], score_40),
                balanced_accuracy_score(labs[mask], fused_score >= 0.5),
                float(lens[mask].mean()),
            ))
        result[family] = rows
    return result


def probe_correctness_per_example(cache, lengths, use_full_tokens: bool = True):
    """Per-example: compute if the probe's sign matches the label."""
    all_lengths = []
    all_correct_fused = []
    all_correct_46 = []
    all_correct_40 = []
    all_families = []
    all_abs_z_fused = []

    for ds_name, entry in cache.items():
        if ds_name not in lengths:
            continue
        len_data = lengths[ds_name]
        labels = entry["labels"]
        l46 = entry["L46"]
        l40 = entry["L40"]
        if labels is None or l46 is None or l40 is None:
            continue

        if use_full_tokens and "full_tokens" in len_data:
            length_arr = len_data["full_tokens"]
        else:
            length_arr = len_data["resp_tokens"]

        family = family_of(ds_name)
        z_fused = probe_z_scores(l46, l40, family)
        z_46 = (l46 - FROZEN_46["probe_logit_mean"][family]) / max(FROZEN_46["probe_logit_sd"][family], 1e-8)
        z_40 = (l40 - FROZEN_40["probe_logit_mean"][family]) / max(FROZEN_40["probe_logit_sd"][family], 1e-8)

        for i in range(len(labels)):
            all_lengths.append(length_arr[i])
            all_correct_fused.append(int((z_fused[i] > 0) == bool(labels[i])))
            all_correct_46.append(int((z_46[i] > 0) == bool(labels[i])))
            all_correct_40.append(int((z_40[i] > 0) == bool(labels[i])))
            all_families.append(family)
            all_abs_z_fused.append(abs(z_fused[i]))

    return (np.array(all_lengths), np.array(all_correct_fused),
            np.array(all_correct_46), np.array(all_correct_40),
            np.array(all_families), np.array(all_abs_z_fused))


def print_correlations(label: str, x: np.ndarray, y: np.ndarray):
    if len(x) < 3:
        print(f"  {label}: n={len(x)} (too few points)")
        return
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    print(f"  {label}: n={len(x):>5}, Pearson r={r_pearson:+.4f} (p={p_pearson:.4f}), "
          f"Spearman ρ={r_spearman:+.4f} (p={p_spearman:.4f})")


def make_plots(per_dataset, per_bin, per_family_bin, save_dir: Path):
    """Generate diagnostic plots."""
    save_dir.mkdir(parents=True, exist_ok=True)

    families_marker = {"qwen": "o", "gemma": "s", "nemotron": "^"}
    families_color = {"qwen": "#1f77b4", "gemma": "#ff7f0e", "nemotron": "#2ca02c"}

    # ---- Plot 1: Per-dataset scatter ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Probe AUROC/BA vs Average Prompt Length (per dataset)", fontsize=13, fontweight="bold")

    metric_keys = [
        ("fused AUROC", 3),
        ("L46 AUROC", 4),
        ("L40 AUROC", 5),
        ("fused BA", 6),
    ]
    for ax, (title, col) in zip(axes.flat, metric_keys):
        for family in ("qwen", "gemma", "nemotron"):
            xs = [r[2] for r in per_dataset if r[1] == family]
            ys = [r[col] for r in per_dataset if r[1] == family]
            if xs:
                ax.scatter(xs, ys, marker=families_marker[family],
                          color=families_color[family], label=family, alpha=0.8, s=50)
        all_xs = [r[2] for r in per_dataset]
        all_ys = [r[col] for r in per_dataset]
        if len(all_xs) > 2:
            r_s, p_s = stats.spearmanr(all_xs, all_ys)
            ax.set_title(f"{title} (ρ={r_s:+.3f}, p={p_s:.3f})")
        else:
            ax.set_title(title)
        ax.set_xlabel("Avg prompt length (tokens)")
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "per_dataset_auroc_vs_length.png", dpi=120)
    print(f"\nSaved {save_dir / 'per_dataset_auroc_vs_length.png'}")

    # ---- Plot 2: Per-bin pooled ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Probe AUROC/BA by Prompt-Length Bin (all datasets pooled)", fontsize=13, fontweight="bold")

    bin_mid = [(r[0] + r[1]) / 2 for r in per_bin]
    ns = [r[2] for r in per_bin]

    ax = axes[0]
    ax.plot(bin_mid, [r[3] for r in per_bin], "o-", label="fused L40+L46", color="black", lw=2)
    ax.plot(bin_mid, [r[4] for r in per_bin], "s--", label="L46", color="#1f77b4", alpha=0.7)
    ax.plot(bin_mid, [r[5] for r in per_bin], "d--", label="L40", color="#ff7f0e", alpha=0.7)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("AUROC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for x, y, n in zip(bin_mid, [r[3] for r in per_bin], ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center")

    ax = axes[1]
    ax.plot(bin_mid, [r[6] for r in per_bin], "o-", color="black", lw=2)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Balanced Accuracy")
    ax.grid(True, alpha=0.3)
    for x, y, n in zip(bin_mid, [r[6] for r in per_bin], ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center")

    if len(bin_mid) > 2:
        r_s_f, p_s_f = stats.spearmanr(bin_mid, [r[3] for r in per_bin])
        ax.set_title(f"BA by length bin (ρ={r_s_f:+.3f}, p={p_s_f:.3f})")

    fig.tight_layout()
    fig.savefig(save_dir / "per_bin_auroc_vs_length.png", dpi=120)
    print(f"Saved {save_dir / 'per_bin_auroc_vs_length.png'}")

    # ---- Plot 3: Per-family bin ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Probe AUROC by Prompt-Length Bin (per model family)", fontsize=13, fontweight="bold")

    for ax, family in zip(axes, ("qwen", "gemma", "nemotron")):
        if family not in per_family_bin:
            ax.set_title(f"{family}: no data")
            continue
        rows = per_family_bin[family]
        mids = [(r[0] + r[1]) / 2 for r in rows]
        ax.plot(mids, [r[3] for r in rows], "o-", label="fused", color="black", lw=2)
        ax.plot(mids, [r[4] for r in rows], "s--", label="L46", color="#1f77b4", alpha=0.7)
        ax.plot(mids, [r[5] for r in rows], "d--", label="L40", color="#ff7f0e", alpha=0.7)
        ns = [r[2] for r in rows]
        for x, y, n in zip(mids, [r[3] for r in rows], ns):
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6),
                        fontsize=6, ha="center")
        if len(mids) > 2:
            r_s, p_s = stats.spearmanr(mids, [r[3] for r in rows])
            ax.set_title(f"{family} (ρ={r_s:+.3f}, p={p_s:.3f})")
        else:
            ax.set_title(family)
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_ylabel("AUROC")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / "per_family_bin_auroc_vs_length.png", dpi=120)
    print(f"Saved {save_dir / 'per_family_bin_auroc_vs_length.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-datasets", action="store_true",
                        help="Skip HF dataset loading; use response tokens from activation files only")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of length bins for per-bin analysis (default: 10)")
    parser.add_argument("--save-dir", type=Path,
                        default=REPO_ROOT / "results/ensemble_gate_eval",
                        help="Directory for output plots and tables")
    parser.add_argument("--use-response-tokens", action="store_true",
                        help="Use response-token count as length metric (instead of full prompt)")
    args = parser.parse_args()

    use_full = not args.use_response_tokens

    print("=" * 70)
    print("Probe AUROC/BA ↔ Prompt Length Correlation")
    print("=" * 70)

    # 1. Load probe logits cache
    print("\n[1] Loading probe logits cache...")
    cache = load_cache()
    print(f"    {len(cache)} datasets with probe logits")

    # 2. Load prompt lengths
    if args.no_datasets:
        print("\n[2] Loading response token counts from activation files (--no-datasets)...")
        lengths = load_prompt_lengths_from_activations(set(cache.keys()))
    else:
        print("\n[2] Loading HF datasets and computing prompt lengths...")
        lengths = load_prompt_lengths_from_datasets(set(cache.keys()))
    print(f"    {len(lengths)} datasets with length data")

    # 3. Per-dataset correlation
    print("\n" + "=" * 70)
    print("[3] Per-dataset correlation (avg length vs AUROC/BA)")
    print("=" * 70)
    per_dataset = compute_per_dataset_metrics(cache, lengths, use_full_tokens=use_full)
    print(f"\n{'dataset':<55} {'fam':>6} {'avg_len':>8} {'sigma_len':>7} "
          f"{'auc_fus':>8} {'auc_L46':>8} {'auc_L40':>8} {'ba_fus':>8} {'n':>5}")
    print("-" * 115)
    for row in per_dataset:
        print(f"{row[0][:54]:<55} {row[1]:>6} {row[2]:>8.0f} {row[3]:>7.0f} "
              f"{row[4]:>8.4f} {row[5]:>8.4f} {row[6]:>8.4f} {row[7]:>8.4f} {row[8]:>5}")

    avg_lens = np.array([r[2] for r in per_dataset])
    auc_fused = np.array([r[4] for r in per_dataset])
    auc_46 = np.array([r[5] for r in per_dataset])
    auc_40 = np.array([r[6] for r in per_dataset])
    ba_fused = np.array([r[7] for r in per_dataset])

    print(f"\n--- Correlations across {len(per_dataset)} datasets ---")
    print_correlations("avg_len ↔ probe AUROC (fused L40+L46)", avg_lens, auc_fused)
    print_correlations("avg_len ↔ probe AUROC (L46 only)", avg_lens, auc_46)
    print_correlations("avg_len ↔ probe AUROC (L40 only)", avg_lens, auc_40)
    print_correlations("avg_len ↔ probe BA   (fused)", avg_lens, ba_fused)

    for family in ("qwen", "gemma", "nemotron"):
        fam_rows = [r for r in per_dataset if r[1] == family]
        if len(fam_rows) >= 3:
            fam_lens = np.array([r[2] for r in fam_rows])
            fam_auc = np.array([r[4] for r in fam_rows])
            print(f"\n  {family} ({len(fam_rows)} datasets):")
            print_correlations(f"    avg_len ↔ probe AUROC", fam_lens, fam_auc)

    # 4. Per-bin analysis (pooled)
    print("\n" + "=" * 70)
    print(f"[4] Per-bin analysis (all datasets pooled, {args.n_bins} bins)")
    print("=" * 70)
    per_bin = compute_per_bin_metrics(cache, lengths, args.n_bins, use_full_tokens=use_full)
    print(f"\n{'bin_start':>8} {'bin_end':>8} {'n':>6} "
          f"{'auc_fus':>8} {'auc_L46':>8} {'auc_L40':>8} {'ba_fus':>8}")
    print("-" * 60)
    for row in per_bin:
        print(f"{row[0]:>8.0f} {row[1]:>8.0f} {row[2]:>6} "
              f"{row[3]:>8.4f} {row[4]:>8.4f} {row[5]:>8.4f} {row[6]:>8.4f}")

    bin_mid = [(r[0] + r[1]) / 2 for r in per_bin]
    if len(bin_mid) > 2:
        print(f"\n--- Trend across bins ---")
        print_correlations("bin_mid ↔ AUROC (fused)", np.array(bin_mid), np.array([r[3] for r in per_bin]))
        print_correlations("bin_mid ↔ BA   (fused)", np.array(bin_mid), np.array([r[6] for r in per_bin]))

    # 5. Per-family bin analysis
    print("\n" + "=" * 70)
    print(f"[5] Per-family bin analysis ({args.n_bins} bins within each family)")
    print("=" * 70)
    per_family_bin = compute_per_family_bin_metrics(cache, lengths, args.n_bins, use_full_tokens=use_full)
    for family, rows in sorted(per_family_bin.items()):
        print(f"\n  {family}:")
        print(f"  {'bin_start':>8} {'bin_end':>8} {'n':>6} "
              f"{'auc_fus':>8} {'auc_L46':>8} {'auc_L40':>8} {'ba_fus':>8} {'avg_len':>8}")
        for row in rows:
            print(f"  {row[0]:>8.0f} {row[1]:>8.0f} {row[2]:>6} "
                  f"{row[3]:>8.4f} {row[4]:>8.4f} {row[5]:>8.4f} {row[6]:>8.4f} {row[7]:>8.0f}")
        mids = [(r[0] + r[1]) / 2 for r in rows]
        if len(mids) > 2:
            print(f"\n  Trend correlations for {family}:")
            print_correlations("    bin_mid ↔ AUROC", np.array(mids), np.array([r[3] for r in rows]))
            print_correlations("    bin_mid ↔ BA", np.array(mids), np.array([r[6] for r in rows]))

    # 6. Per-example correctness correlation
    print("\n" + "=" * 70)
    print("[6] Per-example: probe correctness sign-match vs prompt length")
    print("=" * 70)
    lens, correct_f, correct_46, correct_40, families, abs_z = probe_correctness_per_example(
        cache, lengths, use_full_tokens=use_full)

    print(f"\n  Total examples: {len(lens)}")
    print(f"  Probe fused accuracy: {correct_f.mean():.4f}")
    print(f"  Probe L46 accuracy:   {correct_46.mean():.4f}")
    print(f"  Probe L40 accuracy:   {correct_40.mean():.4f}")
    print()
    print_correlations("length ↔ correctness (fused)", lens, correct_f)
    print_correlations("length ↔ correctness (L46)", lens, correct_46)
    print_correlations("length ↔ correctness (L40)", lens, correct_40)
    print_correlations("length ↔ probe confidence |z_fused|", lens, abs_z)

    print()
    for fam in ("qwen", "gemma", "nemotron"):
        fmask = families == fam
        if fmask.sum() < 50:
            continue
        print(f"  {fam} ({fmask.sum()} examples):")
        print_correlations("    length ↔ correctness (fused)", lens[fmask], correct_f[fmask])
        print_correlations("    length ↔ confidence |z|", lens[fmask], abs_z[fmask])

    # 7. Make plots
    print("\n" + "=" * 70)
    print("[7] Generating plots...")
    print("=" * 70)
    try:
        make_plots(per_dataset, per_bin, per_family_bin, args.save_dir)
    except Exception as e:
        print(f"  WARNING: plot generation failed: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
