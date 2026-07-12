"""
Extract and cache residual-stream activations for probe training.

For one competition dataset, run every row through the generating organism
(base model + LoRA) and cache pooled response-span activations at a sweep of
layers to an .npz on disk. Downstream probe training (linear sweeps, MLPs,
attention poolers) then reads the cache and never touches a GPU again.

The organism loads on a LOCAL GPU by default (the team NDIF key is
submission-only and has no development access); `--remote` switches to one
batched NDIF session should dev access ever be granted.

Usage (on Hábrók, from the repo root):

    sbatch experiments/nonlinear_probe/extract_activations.sh \
        --dataset aletheias-quest/dev-varied-deception-Qwen3.5-27B-a-mo-qwen3.5-27b-1 \
        --limit 32                    # smoke test first

    sbatch experiments/nonlinear_probe/extract_activations.sh --dataset ... \
        --layers ::2                  # every 2nd layer (default)

    sbatch experiments/nonlinear_probe/extract_activations.sh --dataset ... \
        --tokens --layers 40,46       # per-token cache, small explicit layer list

Output: <out-dir>/<dataset-with-__>.npz holding, per layer L:
    mean_L{L}  (N, D) float16   response-span mean-pooled activations
    last_L{L}  (N, D) float16   last-real-token activations
plus `index`, `deceptive` (if the dataset has labels), `response_tokens`,
and a JSON `meta` string (model, lora, layers, tokenization settings).

With `--tokens`, writes <out-dir>/<dataset-with-__>.tokens.npz instead, holding
every response token's activation per layer L:
    tokens_L{L}  (total_tokens, D) float16   every response-token activation
plus `token_offsets` (N + 1,) marking each example's token range in dataset
order, the same `index`/`deceptive`/`response_tokens`/`meta` fields, and
`meta["mode"] == "tokens"`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submission"))

from util import build_model, chat_preprocess, decoder_layers, load_examples  # noqa: E402

# Local dev fallback: the leaderboard injects NDIF_HOST, but a local run needs
# the competition cluster set explicitly unless the env already provides it.
# Matches scripts/dev_leaderboard.py; the tutorials' duckdns host is older.
DEFAULT_NDIF_HOST = "https://aletheias.api.ndif.us"

POOLINGS = ("mean", "last")


def parse_layers(spec: str, num_layers: int) -> list[int]:
    """
    Resolve a layer-selection spec against a model's layer count.

    :param spec: Either slice syntax over range(num_layers), e.g. '::2' or
        '8:56:4', or an explicit comma-separated list like '5,17,29'.
    :param num_layers: Number of decoder layers in the model.
    :return: Sorted list of valid layer indices to extract.
    """
    if ":" in spec:
        parts = [int(part) if part else None for part in spec.split(":")]
        parts += [None] * (3 - len(parts))
        layers = list(range(num_layers))[slice(*parts)]
    else:
        layers = [int(part) for part in spec.split(",")]
    out_of_range = [layer for layer in layers if not 0 <= layer < num_layers]
    if out_of_range:
        raise ValueError(f"layers {out_of_range} out of range for a {num_layers}-layer model")
    return layers


def budget_batches(lengths: list[int], token_budget: int, max_batch: int) -> list[list[int]]:
    """
    Group example positions into length-sorted batches under a token budget.

    Peak GPU memory scales with batch_size x padded_width, so the constraint is
    on the product, not the row count (see llms.txt's OOM guidance).

    :param lengths: Token count of each example, in dataset order.
    :param token_budget: Maximum batch_size x padded_width per batch.
    :param max_batch: Hard cap on rows per batch regardless of width.
    :return: List of batches, each a list of positions into `lengths`.
    """
    order = sorted(range(len(lengths)), key=lambda position: lengths[position])
    batches: list[list[int]] = []
    current: list[int] = []
    for position in order:
        # Ascending length order, so the newest element sets the padded width.
        width = lengths[position]
        if current and ((len(current) + 1) * width > token_budget or len(current) >= max_batch):
            batches.append(current)
            current = []
        current.append(position)
    if current:
        batches.append(current)
    return batches


def collate_batch(
        token_lists: list[list[int]],
        spans: list[tuple[int, int]],
        pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Right-pad one batch and build its response-span pooling tensors.

    :param token_lists: Token ids of each example in the batch.
    :param spans: (start, end) assistant-response span per example.
    :param pad_id: Token id used for right padding.
    :return: Tuple of (input_ids, attention_mask, response_mask, last_token_idx),
        where response_mask is (B, T) float marking response tokens and
        last_token_idx is (B,) long with each row's final real token position.
    """
    width = max(len(tokens) for tokens in token_lists)
    batch_size = len(token_lists)
    input_ids = torch.tensor([tokens + [pad_id] * (width - len(tokens)) for tokens in token_lists])
    attention_mask = torch.tensor([[1] * len(tokens) + [0] * (width - len(tokens))
                                   for tokens in token_lists])
    response_mask = torch.zeros((batch_size, width))
    last_token_idx = torch.zeros(batch_size, dtype=torch.long)
    for row, (span_start, span_end) in enumerate(spans):
        response_mask[row, span_start:span_end] = 1.0
        last_token_idx[row] = max(span_end - 1, 0)
    return input_ids, attention_mask, response_mask, last_token_idx


def extract_pooled_activations(
        model: Any,
        layers: list[int],
        batches: list[list[int]],
        token_lists: list[list[int]],
        spans: list[tuple[int, int]],
        pad_id: int,
        remote: bool = False,
) -> list[torch.Tensor]:
    """
    Run every batch through the model and pool activations at the chosen layers.

    Inside each trace, reads the residual stream at every requested layer,
    pools it two ways over the response span (mean and last real token), and
    stops the forward early after the deepest requested layer. With
    `remote=True` all batches are bundled into one NDIF session (one queue
    wait); locally each trace just executes on the dispatched model.

    :param model: nnsight model handle (local GPU or NDIF-backed).
    :param layers: Decoder layer indices to extract.
    :param batches: Batches as lists of positions into `token_lists`.
    :param token_lists: Token ids of every example.
    :param spans: (start, end) response span of every example.
    :param pad_id: Padding token id.
    :param remote: Run on NDIF instead of a local GPU (needs a dev-access key).
    :return: One (num_layers, num_poolings, N, D) float16 CPU tensor, rows in
        batch order (all of batches[0], then batches[1], ...).
    """
    layer_modules = decoder_layers(model)
    session = model.session(remote=True) if remote else contextlib.nullcontext()
    with session:
        # The session body executes remotely: only values that flow into the
        # final .save() survive it, so per-batch results are collected in a
        # list created INSIDE the session and concatenated once at the end
        # (the same shape as util.run_full_session).
        pieces = []
        for batch_positions in batches:
            input_ids, attention_mask, response_mask, last_token_idx = collate_batch(
                [token_lists[position] for position in batch_positions],
                [spans[position] for position in batch_positions],
                pad_id,
            )
            with model.trace({"input_ids": input_ids, "attention_mask": attention_mask}) as tracer:
                pooled_per_layer = []
                for layer in layers:
                    # transformers 5.x decoder layers return the tensor directly;
                    # older versions return a tuple with the hidden state first.
                    hidden = layer_modules[layer].output
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                    mask = response_mask.to(hidden.device).unsqueeze(-1)
                    denom = response_mask.to(hidden.device).sum(1, keepdim=True).clamp(min=1)
                    mean_pooled = (hidden * mask).sum(1) / denom
                    last_pooled = hidden[torch.arange(hidden.shape[0]),
                                         last_token_idx.to(hidden.device)]
                    # Layers are sharded across GPUs on big deployments, so move
                    # each pooled pair to CPU before stacking across layers.
                    pooled_per_layer.append(
                        torch.stack([mean_pooled, last_pooled], 0).to(torch.float16).detach().cpu())
                stacked = torch.stack(pooled_per_layer, 0).save()
                # Skip every layer past max(layers) and the LM head.
                tracer.stop()
            pieces.append(stacked)
        all_features = torch.cat(pieces, dim=2)
        if remote:
            all_features = all_features.save()
    return all_features


def extract_token_activations(
        model: Any,
        layers: list[int],
        batches: list[list[int]],
        token_lists: list[list[int]],
        spans: list[tuple[int, int]],
        pad_id: int,
        remote: bool = False,
) -> torch.Tensor:
    """
    Run every batch through the model and collect every response token's
    activation at the chosen layers, with no pooling.

    Mirrors `extract_pooled_activations`'s session/trace/tracer.stop structure
    and remote-save discipline (only values flowing into the final `.save()`
    survive a remote session). Inside each trace, reads the residual stream at
    every requested layer and selects the response-span tokens in row-major
    order, then stops the forward early after the deepest requested layer.

    :param model: nnsight model handle (local GPU or NDIF-backed).
    :param layers: Decoder layer indices to extract.
    :param batches: Batches as lists of positions into `token_lists`.
    :param token_lists: Token ids of every example.
    :param spans: (start, end) response span of every example.
    :param pad_id: Padding token id.
    :param remote: Run on NDIF instead of a local GPU (needs a dev-access key).
    :return: One (num_layers, total_tokens, hidden) float16 CPU tensor, tokens
        in batch-traversal order (all of batches[0]'s rows' response tokens in
        row order, then batches[1]'s, ...).
    """
    layer_modules = decoder_layers(model)
    session = model.session(remote=True) if remote else contextlib.nullcontext()
    with session:
        # Same discipline as extract_pooled_activations: per-batch results are
        # collected in a list created INSIDE the session and concatenated once
        # at the end.
        pieces = []
        for batch_positions in batches:
            input_ids, attention_mask, response_mask, _ = collate_batch(
                [token_lists[position] for position in batch_positions],
                [spans[position] for position in batch_positions],
                pad_id,
            )
            with model.trace({"input_ids": input_ids, "attention_mask": attention_mask}) as tracer:
                selected_per_layer = []
                for layer in layers:
                    # transformers 5.x decoder layers return the tensor directly;
                    # older versions return a tuple with the hidden state first.
                    hidden = layer_modules[layer].output
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                    mask_bool = response_mask.bool().to(hidden.device)
                    selected = hidden[mask_bool]
                    # Layers are sharded across GPUs on big deployments, so move
                    # each selection to CPU before stacking across layers.
                    selected_per_layer.append(selected.to(torch.float16).detach().cpu())
                stacked = torch.stack(selected_per_layer, 0).save()
                # Skip every layer past max(layers) and the LM head.
                tracer.stop()
            pieces.append(stacked)
        all_features = torch.cat(pieces, dim=1)
        if remote:
            all_features = all_features.save()
    return all_features


def assemble_token_features(
        flat_features: np.ndarray,
        batches: list[list[int]],
        span_lengths: list[int],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Re-order per-token activations from batch-traversal order to dataset
    order and compute each example's token offsets.

    :param flat_features: (num_layers, total_tokens, hidden) array with the
        token axis in batch-traversal order, as produced by
        `extract_token_activations` (all of batches[0]'s rows' response
        tokens in row order, then batches[1]'s, ...).
    :param batches: Batches as lists of positions into the original dataset
        order, the same batches passed to `extract_token_activations`.
    :param span_lengths: Response-token count of every example, indexed by
        its position in the original dataset order.
    :return: Tuple of (per-layer dataset-ordered arrays keyed by their
        position in `flat_features`'s layer axis, `token_offsets`), where
        `token_offsets` is (N + 1,) int64 and example i's tokens are
        `layer_features[token_offsets[i]:token_offsets[i + 1]]`.
    """
    flat_positions = [position for batch_positions in batches for position in batch_positions]
    traversal_lengths = [span_lengths[position] for position in flat_positions]
    traversal_offsets = np.concatenate([[0], np.cumsum(traversal_lengths)])
    traversal_index_by_position = {position: traversal_index
                                   for traversal_index, position in enumerate(flat_positions)}
    token_offsets = np.concatenate([[0], np.cumsum(span_lengths)]).astype(np.int64)
    num_layers = flat_features.shape[0]
    dataset_order_features: dict[int, np.ndarray] = {}
    for layer_position in range(num_layers):
        layer_pieces = []
        for position in range(len(span_lengths)):
            traversal_index = traversal_index_by_position[position]
            start = traversal_offsets[traversal_index]
            end = traversal_offsets[traversal_index + 1]
            layer_pieces.append(flat_features[layer_position, start:end])
        dataset_order_features[layer_position] = np.concatenate(layer_pieces, axis=0)
    return dataset_order_features, token_offsets


def load_labels(dataset_name: str, examples: Any, num_rows: int) -> Any:
    """
    Fetch `deceptive` labels for a dev dataset, joined by `index`.

    Dev inputs repos don't carry the label column; it lives in the sibling
    `<dataset>-labels` repo (the same split dry.yaml points labels_uri at).
    Private leaderboard datasets have no labels repo, so failure to load one
    is expected and returns None.

    :param dataset_name: HF id of the inputs dataset.
    :param examples: The loaded inputs dataset (provides the `index` order).
    :param num_rows: Number of rows actually extracted.
    :return: (num_rows,) bool array in extraction order, or None if no labels.
    """
    if "deceptive" in examples.column_names:
        return np.asarray(examples["deceptive"][:num_rows], dtype=bool)
    try:
        labels_ds = load_dataset(dataset_name + "-labels", split="test")
    except Exception as error:
        print(f"no labels repo for {dataset_name} ({type(error).__name__}); caching unlabeled")
        return None
    by_index = dict(zip(labels_ds["index"], labels_ds["deceptive"]))
    return np.asarray([by_index[i] for i in examples["index"][:num_rows]], dtype=bool)


def main() -> None:
    """
    Parse CLI arguments, extract activations for one dataset, and write the cache.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", required=True, help="HF dataset id (single test split)")
    parser.add_argument("--layers", default="::2",
                        help="slice 'start:stop:step' over layer indices, or comma list")
    parser.add_argument("--max-len", type=int, default=512,
                        help="token cap per example (left-trimmed so the response survives)")
    parser.add_argument("--token-budget", type=int, default=2560,
                        help="max batch_size x padded_width per trace (OOM guard)")
    parser.add_argument("--max-batch", type=int, default=32, help="hard cap on rows per trace")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N rows (smoke test)")
    parser.add_argument("--out-dir", default="results/whitebox/activations")
    parser.add_argument("--force", action="store_true", help="overwrite an existing cache file")
    parser.add_argument("--remote", action="store_true",
                        help="run on NDIF instead of a local GPU (requires a dev-access "
                             "key; the team key is submission-only)")
    parser.add_argument("--tokens", action="store_true",
                        help="store per-token response activations instead of pooled "
                             "vectors (use with a small explicit --layers list, e.g. "
                             "--layers 40,46)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".limit{args.limit}" if args.limit else ""
    mode_suffix = ".tokens" if args.tokens else ""
    out_path = out_dir / (args.dataset.replace("/", "__") + mode_suffix + suffix + ".npz")
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists; use --force to redo")
        return

    if args.remote and not os.environ.get("NDIF_HOST"):
        os.environ["NDIF_HOST"] = DEFAULT_NDIF_HOST
        print(f"NDIF_HOST unset; defaulting to {DEFAULT_NDIF_HOST}")

    examples = load_examples(args.dataset)
    num_rows = len(examples) if args.limit is None else min(args.limit, len(examples))
    model_id, lora_id = examples[0]["model"], examples[0].get("lora")
    print(f"{args.dataset}: {num_rows}/{len(examples)} rows, model={model_id}, lora={lora_id}")

    model_kwargs = {} if args.remote else {"device_map": "auto", "dtype": torch.bfloat16}
    model = build_model(model_id, lora_id, **model_kwargs)
    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    token_lists: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    for messages in examples["messages"][:num_rows]:
        token_ids, span = chat_preprocess(messages, tokenizer, args.max_len)
        token_lists.append(token_ids)
        spans.append(span)

    config = model.config
    num_layers = getattr(config, "text_config", config).num_hidden_layers
    layers = parse_layers(args.layers, num_layers)
    batches = budget_batches([len(tokens) for tokens in token_lists],
                             args.token_budget, args.max_batch)
    print(f"{num_layers} layers -> extracting {len(layers)}: {layers}")
    print(f"{len(batches)} batches (token budget {args.token_budget}, "
          f"max batch {args.max_batch})")

    start_time = time.time()
    if args.tokens:
        all_features = extract_token_activations(model, layers, batches, token_lists, spans,
                                                  pad_id, remote=args.remote)
    else:
        all_features = extract_pooled_activations(model, layers, batches, token_lists, spans,
                                                  pad_id, remote=args.remote)
    print(f"extraction done in {time.time() - start_time:.1f}s")

    span_lengths = [span_end - span_start for span_start, span_end in spans]
    labels = load_labels(args.dataset, examples, num_rows)
    meta = dict(
        dataset=args.dataset, model=model_id, lora=lora_id, layers=layers,
        poolings=list(POOLINGS), max_len=args.max_len, n=num_rows,
        token_budget=args.token_budget,
        extracted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if args.tokens:
        layer_features, token_offsets = assemble_token_features(
            all_features.cpu().numpy(), batches, span_lengths)
        cache: dict[str, np.ndarray] = {
            f"tokens_L{layer}": layer_features[layer_position].astype(np.float16)
            for layer_position, layer in enumerate(layers)
        }
        cache["token_offsets"] = token_offsets
        cache["index"] = np.asarray(examples["index"][:num_rows])
        cache["response_tokens"] = np.asarray(span_lengths, dtype=np.int32)
        if labels is not None:
            cache["deceptive"] = labels
        cache["meta"] = np.asarray(json.dumps({**meta, "mode": "tokens"}))
        np.savez(out_path, **cache)
    else:
        flat_positions = [position for batch_positions in batches for position in batch_positions]
        sorted_features = all_features.cpu().numpy()
        hidden_dim = sorted_features.shape[-1]
        features = {f"{pooling}_L{layer}": np.empty((num_rows, hidden_dim), dtype=np.float16)
                    for layer in layers for pooling in POOLINGS}
        for layer_pos, layer in enumerate(layers):
            for pool_pos, pooling in enumerate(POOLINGS):
                features[f"{pooling}_L{layer}"][flat_positions] = sorted_features[layer_pos, pool_pos]

        cache = dict(features)
        cache["index"] = np.asarray(examples["index"][:num_rows])
        cache["response_tokens"] = np.asarray(span_lengths, dtype=np.int32)
        if labels is not None:
            cache["deceptive"] = labels
        cache["meta"] = np.asarray(json.dumps(meta))
        np.savez_compressed(out_path, **cache)

    size_mb = out_path.stat().st_size / 1e6
    label_note = (f", {int(cache['deceptive'].sum())}/{num_rows} deceptive"
                  if "deceptive" in cache else ", no labels")
    print(f"wrote {out_path} ({size_mb:.1f} MB, {num_rows} rows, "
          f"{len(layers)} layers{label_note})")


if __name__ == "__main__":
    main()
