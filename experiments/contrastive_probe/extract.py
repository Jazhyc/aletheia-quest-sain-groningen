"""
Extract contrastive activations from dev datasets.

For each example in a dataset, runs TWO forward passes:
  Pass 1: original conversation (with deceptive system prompt)
  Pass 2: neutral conversation (deceptive system prompt removed)

Saves both activation sets and the per-token difference to .npz files.
One NDIF session per dataset. Uses the same L46 extraction as the existing probes.
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submission"))
import util  # noqa: E402


def strip_system_prompt(messages: list[dict]) -> list[dict]:
    """Remove the first message if it's a system role (the deceptive prompt)."""
    msgs = copy.deepcopy(messages)
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        msgs.pop(0)
    return msgs


def extract_dataset(dataset_name: str, layer: int = 46, max_len: int = 512,
                    token_budget: int = 2560, max_batch_rows: int = 32,
                    limit: int | None = None, out_dir: Path | None = None,
                    max_attempts: int = 4):
    """Extract contrastive activations for one dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="test")
    if limit:
        ds = ds.select(range(limit))
    print(f"Dataset: {dataset_name} ({len(ds)} examples)")

    model_id = ds[0]["model"]
    lora = ds[0].get("lora", None)
    print(f"  model={model_id}  lora={lora}")

    model = util.build_model(model_id, lora)
    tokenizer = model.tokenizer
    PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    layers = util.decoder_layers(model)
    layer_idx = min(layer, len(layers) - 1)
    print(f"  layer={layer_idx} (of {len(layers)})")

    # Tokenize both original and neutral conversations
    orig_tokens, orig_spans = [], []
    neut_tokens, neut_spans = [], []
    indices_list = []

    for ex in ds:
        idx = ex.get("index", len(indices_list))
        indices_list.append(idx)

        tids, span = util.chat_preprocess(ex["messages"], tokenizer, max_len=max_len)
        orig_tokens.append(tids)
        orig_spans.append(span)

        neut_msgs = strip_system_prompt(ex["messages"])
        ntids, nspan = util.chat_preprocess(neut_msgs, tokenizer, max_len=max_len)
        neut_tokens.append(ntids)
        neut_spans.append(nspan)

    # Build batches (sorted by length, same batching for both passes)
    lengths = [len(t) for t in orig_tokens]
    order = sorted(range(len(lengths)), key=lambda p: lengths[p])
    batches = []
    current = []
    for pos in order:
        if current and ((len(current) + 1) * lengths[pos] > token_budget
                        or len(current) >= max_batch_rows):
            batches.append(current)
            current = []
        current.append(pos)
    if current:
        batches.append(current)
    print(f"  {len(batches)} batches")

    # ---- Pass 1: original (deceptive) ----
    print("  Pass 1: deceptive...")
    t0 = time.time()

    def extract_pass(token_lists, spans_list, label):
        with model.session(remote=True):
            pieces = []
            for bp in batches:
                bt = [token_lists[p] for p in bp]
                bs = [spans_list[p] for p in bp]
                width = max(len(t) for t in bt)
                rows = len(bt)
                ids = torch.full((rows, width), PAD_ID, dtype=torch.long)
                am = torch.zeros(rows, width, dtype=torch.long)
                rm = torch.zeros(rows, width, dtype=torch.bool)
                for r, (toks, (s, e)) in enumerate(zip(bt, bs)):
                    ids[r, :len(toks)] = torch.tensor(toks)
                    am[r, :len(toks)] = 1
                    rm[r, s:e] = True
                with model.trace({"input_ids": ids, "attention_mask": am}) as tracer:
                    hid = layers[layer_idx].output
                    if isinstance(hid, tuple):
                        hid = hid[0]
                    mb = rm.to(hid.device)
                    sel = hid[mb].to(torch.float16).detach().cpu().save()
                    tracer.stop()
                pieces.append(sel)
            flat = torch.cat(pieces, dim=0)
            flat = flat.save()
        raw = flat.cpu().numpy().astype(np.float32)
        finfo = np.finfo(np.float16)
        return np.clip(raw, finfo.min, finfo.max)

    # Retry loop
    for attempt in range(1, max_attempts + 1):
        try:
            deceptive_flat = extract_pass(orig_tokens, orig_spans, "deceptive")
            break
        except Exception as e:
            if attempt >= max_attempts:
                raise
            wait = min(30, 2 ** attempt)
            print(f"    transient error attempt {attempt}: {type(e).__name__}; retrying in {wait}s")
            time.sleep(wait)

    print(f"    done in {time.time() - t0:.0f}s, shape={deceptive_flat.shape}")

    # ---- Pass 2: neutral ----
    print("  Pass 2: neutral...")
    t0 = time.time()

    for attempt in range(1, max_attempts + 1):
        try:
            neutral_flat = extract_pass(neut_tokens, neut_spans, "neutral")
            break
        except Exception as e:
            if attempt >= max_attempts:
                raise
            wait = min(30, 2 ** attempt)
            print(f"    transient error attempt {attempt}: {type(e).__name__}; retrying in {wait}s")
            time.sleep(wait)

    print(f"    done in {time.time() - t0:.0f}s, shape={neutral_flat.shape}")

    # ---- Compute offsets ----
    batch_order = [p for batch in batches for p in batch]
    orig_lengths = [orig_spans[p][1] - orig_spans[p][0] for p in batch_order]
    orig_piece_offsets = np.cumsum([0] + orig_lengths).astype(np.int64)
    slot_of = {p: s for s, p in enumerate(batch_order)}

    neut_lengths = [neut_spans[p][1] - neut_spans[p][0] for p in batch_order]
    neut_piece_offsets = np.cumsum([0] + neut_lengths).astype(np.int64)

    # Reorder to dataset order and compute differences
    diff_tokens = []
    diff_arrays = []
    row_offsets = [0]
    row_indices = []
    row_deceptive = []

    # Load labels for dev datasets
    try:
        labels_ds = load_dataset(f"{dataset_name}-labels", split="test")
        label_map = {ex["index"]: ex["deceptive"] for ex in labels_ds}
    except Exception:
        label_map = {}

    for p in range(len(orig_spans)):
        ds_len = orig_spans[p][1] - orig_spans[p][0]
        ns_len = neut_spans[p][1] - neut_spans[p][0]
        ml = min(ds_len, ns_len)

        d_s = int(orig_piece_offsets[slot_of[p]])
        n_s = int(neut_piece_offsets[slot_of[p]])

        if ml > 0:
            diff = deceptive_flat[d_s:d_s + ml] - neutral_flat[n_s:n_s + ml]
            diff_arrays.append(diff)
            diff_tokens.append(ml)
            row_offsets.append(row_offsets[-1] + ml)
        else:
            diff_tokens.append(0)
            row_offsets.append(row_offsets[-1])

        row_indices.append(indices_list[p])
        row_deceptive.append(label_map.get(indices_list[p], -1))

    diff_flat = np.concatenate(diff_arrays) if diff_arrays else np.zeros((0, deceptive_flat.shape[1]), dtype=np.float32)
    offsets_arr = np.array(row_offsets, dtype=np.int64)

    # ---- Save ----
    if out_dir is None:
        out_dir = REPO_ROOT / "results" / "contrastive" / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = dataset_name.replace("/", "_")
    out_path = out_dir / f"{safe_name}.npz"
    np.savez_compressed(
        out_path,
        diff_flat=diff_flat,
        offsets=offsets_arr,
        indices=np.array(row_indices),
        deceptive=np.array(row_deceptive),
        hidden_dim=deceptive_flat.shape[1],
        layer=layer,
    )
    print(f"  Saved to {out_path}")
    print(f"  {len(row_indices)} examples, {diff_flat.shape[0]} diff tokens")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract contrastive activations")
    parser.add_argument("--dataset", default=None, help="HF dataset name (required unless --all-dev)")
    parser.add_argument("--layer", type=int, default=46)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--all-dev", action="store_true",
                        help="Extract all 22 dev datasets")
    args = parser.parse_args()

    if not args.all_dev and not args.dataset:
        parser.error("--dataset is required unless --all-dev is set")

    if args.all_dev:
        # All dev datasets from the nonlinear probe sweep
        DEV_DATASETS = [
            # Qwen instructed
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-None",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-b-mo-qwen3.5-27b",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-c-mo-qwen3.5-27b",
            "aletheias-quest/dev-instructed-deception-Qwen3.5-27B-g-st-qwen3.5-27b",
            # Qwen varied
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-None",
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1",
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-3",
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-4",
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-5",
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-6",
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-7",
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-b-mo-qwen3.5-27b",
            "aletheias-quest/dev-varied-deception-Qwen3.5-27B-c-mo-qwen3.5-27b",
            # gemma instructed
            "aletheias-quest/dev-instructed-deception-gemma-3-27b-it-None",
            "aletheias-quest/dev-instructed-deception-gemma-3-27b-it-g-st-gemma-3-27b-it-2",
            "aletheias-quest/dev-instructed-deception-gemma-3-27b-it-s-mo-gemma-3-27b-it",
            # Nemotron
            "aletheias-quest/dev-instructed-deception-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-None",
        ]
        for ds_name in DEV_DATASETS:
            try:
                extract_dataset(ds_name, layer=args.layer, limit=args.limit, out_dir=args.out_dir)
            except Exception as e:
                print(f"FAILED {ds_name}: {type(e).__name__}: {e}")
    else:
        extract_dataset(args.dataset, layer=args.layer, limit=args.limit, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
