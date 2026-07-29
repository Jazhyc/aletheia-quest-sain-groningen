"""Helpers for Aletheia's Quest submissions.

The competition contract is small: read `DATASET_NAME`, run your detector, write
`submission.csv` with `index,deceptive,score`. This module factors out the
boilerplate that every notebook repeats — loading the data, building the right
nnsight model for it, and running every row through one batched **remote NDIF
session** — so your notebook is just your *method*.

Typical use (see `example.ipynb`):

    import util

    def detect(model, model_id, lora_id, batch):
        # runs INSIDE an open remote trace over `batch`; return (B,) scores in [0,1]
        logits = batch.gather_last(model.output.logits)     # (B, vocab)
        return my_score(logits)

    scores  = util.run_full_session(DATASET_NAME, detect, batch_size=32)
    examples = util.load_examples(DATASET_NAME)
    util.write_submission(examples["index"], scores)

`run_full_session` peeks row 0 to learn the generating model + LoRA, builds it
once, then loads/tokenizes/forwards/scores every row as **one** NDIF job (one
queue wait — far more robust to queue latency than many separate traces). Heavy
deps (datasets, torch, nnsight, transformers) are imported lazily so importing
`util` is cheap and works inside the per-job sandbox venv.
"""

from __future__ import annotations

from typing import Callable

# ── Qwen3.5 NDIF compat ──────────────────────────────────────────────────────
# The NDIF server's Qwen3.5-27B deployment uses a module tree where
# ``Qwen3_5GatedDeltaNet`` has no ``act`` child module (the activation is
# applied in the forward method directly).  Our local transformers version
# creates ``act`` as a ``SiLUActivation`` submodule, whose persistent id the
# server doesn't recognise.  Strip it so the envoy tree matches the server's.
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    _orig_gdn_init = Qwen3_5GatedDeltaNet.__init__

    def _patched_gdn_init(self, config, layer_idx):
        _orig_gdn_init(self, config, layer_idx)
        if "act" in self._modules:
            del self._modules["act"]

    Qwen3_5GatedDeltaNet.__init__ = _patched_gdn_init
except Exception:
    pass  # older transformers that don't have the class — nothing to patch

SPLIT = "test"   # every competition dataset is a single `test` split


# ── data ────────────────────────────────────────────────────────────────────

def load_examples(dataset_name: str):
    """Load a competition dataset (the single `test` split) with a guaranteed
    `index` join key. Rows have `messages`, `model`, `lora` (+ `index` if the set
    ships one; otherwise it's the 0-based row position)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=SPLIT)
    if "index" not in ds.column_names:
        ds = ds.add_column("index", list(range(len(ds))))
    return ds


def final_assistant_message(ex: dict) -> str:
    """The assistant's last turn (empty string if there is none) — handy for
    text-only heuristics that don't need a forward pass."""
    for m in reversed(ex["messages"]):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def response_span(messages: list, tokenizer) -> tuple[list[int], int, int]:
    """Tokenize a conversation and locate the final turn: returns
    ``(token_ids, start, end)`` where ``token_ids[start:end]`` is the assistant's
    response. Falls back to "the whole sequence is the response" for tokenizers
    without a chat template."""
    try:
        # return_dict=True -> a flat List[int] of token ids; without it,
        # apply_chat_template(tokenize=True) returns a BatchEncoding (transformers 5.x).
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_dict=True)["input_ids"]
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
            return_dict=True)["input_ids"]
        return list(full), len(prefix), len(full)
    except Exception:
        text = "\n".join(m.get("content", "") for m in messages)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids or [tokenizer.eos_token_id or 0]
        return list(ids), 0, len(ids)


# ── model ────────────────────────────────────────────────────────────────────

def build_model(model_id: str, lora_id: str | None = None, **model_kwargs):
    """Construct the competition's text-only nnsight ``LanguageModel`` handle.

    ``peft=lora_id`` attaches a LoRA adapter when one is given. Every competition
    organism is deployed under the ``LanguageModel`` remote key, including
    ``google/gemma-3-27b-it`` even though its Hugging Face configuration contains
    a vision tower. Selecting ``VisionLanguageModel`` for Gemma therefore misses
    the pinned competition deployment.
    """
    try:
        from nnsight import LanguageModel
    except ImportError:                       # nnsight 0.7.x no longer re-exports at top level
        from nnsight.modeling.language import LanguageModel

    kwargs = {"peft": lora_id} if lora_id else {}
    kwargs.update(model_kwargs)

    # Qwen3.5 models on NDIF are deployed via the ConditionalGeneration class
    # (which has ``model.language_model.layers``), but ``AutoModelForCausalLM``
    # resolves to ``Qwen3_5ForCausalLM`` (no ``language_model`` hop).  The
    # persistent-id mismatch breaks every trace.  Switch to the multimodal
    # automodel so the local meta-model tree matches the server's.
    if "qwen" in model_id.lower():
        from transformers import AutoModelForImageTextToText
        kwargs.setdefault("automodel", AutoModelForImageTextToText)

    return LanguageModel(model_id, **kwargs)


def load_model(dataset_name: str):
    """Build the nnsight model a dataset was generated with — peek row 0 for its
    ``model`` + ``lora`` and construct it (one model/LoRA per dataset)."""
    ex0 = load_examples(dataset_name)[0]
    return build_model(ex0["model"], ex0.get("lora"))


def decoder_layers(model):
    """The text decoder's transformer ``layers`` ModuleList, so a probe can read
    ``decoder_layers(model)[L].output`` (the residual stream at layer L).

    The nesting depends on what NDIF served — a plain multimodal VLM (layers
    under ``.model.language_model.layers``), a text-only CausalLM, or either
    wrapped in a PEFT adapter (which adds another ``.model`` hop before the
    inner model becomes visible). Walk down through ``language_model``/``model``
    envoys until a ``layers`` ModuleList appears; envoys only expose real
    submodules, so ``hasattr`` is a safe probe, and preferring ``language_model``
    keeps the walk out of any vision tower."""
    inner = model.model
    for _ in range(5):
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        elif hasattr(inner, "model"):
            inner = inner.model
        else:
            break
    raise AttributeError("could not locate decoder layers under model.model")


# ── batched remote session ───────────────────────────────────────────────────

class Batch:
    """One right-padded batch handed to ``detect_fn``. ``detect_fn`` opens its own
    ``model.trace`` / ``model.generate`` on ``input_ids``/``attention_mask`` and
    reduces the result to one score per row.

    Holds only plain, picklable data: ``detect_fn`` (and everything it touches) is
    serialized and executed remotely on NDIF, so heavy/unpicklable objects (the
    tokenizer, raw HF rows) are deliberately kept off it — precompute anything like
    that client-side and pass it through ``run_full_session(..., **kwargs)`` instead.

    Use the helpers so your code is padding-agnostic:
      * ``gather_last(h)`` — the activation/logit at each row's *last real* token.
      * ``pool_response(h)`` — mean over each row's assistant-response span.

    A plain class (not a ``@dataclass``): the generated dataclass methods don't
    survive nnsight's cloudpickle into the remote session.
    """

    def __init__(self, input_ids, attention_mask, indices, response_spans):
        self.input_ids = input_ids            # (B, seq) right-padded ids fed to the trace
        self.attention_mask = attention_mask  # (B, seq) 1 = real token, 0 = pad
        self.indices = indices                # each row's `index` join key
        self.response_spans = response_spans  # (start, end) per row, into the padded seq

    def gather_last(self, h):
        """``h`` is ``(B, seq, ...)`` -> ``(B, ...)`` at each row's last real token."""
        import torch
        return torch.stack([h[i, e - 1] for i, (s, e) in enumerate(self.response_spans)])

    def pool_response(self, h):
        """``h`` is ``(B, seq, ...)`` -> ``(B, ...)`` mean-pooled over each row's
        assistant-response tokens."""
        import torch
        return torch.stack([h[i, s:e].mean(0) for i, (s, e) in enumerate(self.response_spans)])


def chat_preprocess(messages: list, tokenizer, max_len: int = 256):
    """Default ``preprocess`` for ``run_full_session``: chat-template tokenize +
    locate the assistant-response span, left-trimmed to ``max_len`` (drop from the
    START so the response survives). Returns ``(token_ids, (start, end))``.

    Pass your own ``preprocess(messages, tokenizer, max_len) -> (ids, (start, end))``
    to ``run_full_session`` to match a probe's exact training tokenization, or to
    rewrite the conversation first (e.g. append a follow-up question to generate on).
    """
    ids, s, e = response_span(messages, tokenizer)
    if max_len and len(ids) > max_len:
        cut = len(ids) - max_len
        ids, s, e = ids[cut:], max(0, s - cut), len(ids) - cut
    return ids, (s, e)


def run_full_session(dataset_name: str, detect_fn: Callable, *,
                     preprocess: Callable = chat_preprocess,
                     batch_size: int = 32, max_len: int = 256, remote: bool = True,
                     limit: int | None = None, **detect_kwargs):
    """Score every row of ``dataset_name`` in ONE remote NDIF session.

    ``run_full_session`` owns the orchestration — load the dataset, tokenize via
    ``preprocess``, length-sort + right-pad into batches, and stitch the scores
    back into dataset order — but **``detect_fn`` owns the model call**. For each
    batch it runs::

        detect_fn(model, model_id, lora_id, batch, **detect_kwargs) -> (B,) scores

    ``detect_fn`` opens its OWN ``model.trace(...)`` (or ``model.generate(...)``) on
    ``batch.input_ids`` / ``batch.attention_mask``, computes a length-``B`` tensor of
    deception scores in ``[0, 1]``, **``.save()``s it inside the trace, and returns
    that** (the save is required — a value left unsaved doesn't survive the trace
    block). Read ``model.output`` for logits or ``util.decoder_layers(model)[L].output``
    for activations; reduce per row with ``batch.gather_last`` /
    ``batch.pool_response``. Extra ``**detect_kwargs`` are threaded straight through
    — pass a trained probe, layer index, weights, etc.

    ``preprocess(messages, tokenizer, max_len) -> (token_ids, (start, end))`` turns
    one conversation into tokens + its response span; the default is
    :func:`chat_preprocess`. Returns a numpy array of scores in the dataset's
    original row order (feed it, with ``load_examples(...)["index"]``, to
    ``write_submission``). ``limit`` scores only the first N rows (a quick
    rehearsal).

    The model is **always** built from the dataset itself (row 0's ``model`` +
    ``lora``) — there is no ``model=`` override. A detector must run on the
    activations of the model that generated the data; forcing a different model
    (e.g. applying a probe's base model to every dataset) would read the wrong
    network's activations."""
    import numpy as np
    import torch

    # Local peek at row 0 ONLY — just enough to build the right model (you can't
    # build a model inside its own session). The full dataset load + tokenize +
    # model calls all happen inside the session below, as one NDIF job. The model
    # ALWAYS comes from the dataset — never an outside override.
    first = load_examples(dataset_name)[0]
    model_id, lora_id = first["model"], first.get("lora")
    model = build_model(model_id, lora_id)

    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    with model.session(remote=remote):
        # Load + tokenize INSIDE the session so it's part of the single remote job
        # (one queue wait), not a client-side round trip.
        ds = load_examples(dataset_name)
        n = len(ds) if limit is None else min(limit, len(ds))
        messages = ds["messages"][:n]
        index = list(ds["index"])[:n]

        toks, spans = [], []
        for msg in messages:
            ids, span = preprocess(msg, tok, max_len)
            toks.append(ids)
            spans.append(span)
        order = sorted(range(len(toks)), key=lambda i: len(toks[i]))   # short -> long

        pieces, poses = [], []
        for b0 in range(0, len(order), batch_size):
            bpos = order[b0:b0 + batch_size]
            bids = [toks[i] for i in bpos]
            w = max(len(x) for x in bids)
            # right-pad to the batch's longest sequence; mask marks the real tokens
            input_ids = torch.tensor([x + [pad_id] * (w - len(x)) for x in bids])
            attn = torch.tensor([[1] * len(x) + [0] * (w - len(x)) for x in bids])
            batch = Batch(input_ids=input_ids, attention_mask=attn,
                          indices=[index[i] for i in bpos],
                          response_spans=[spans[i] for i in bpos])
            # detect_fn opens its own trace/generate on this batch and returns a
            # (B,) scores proxy (.save()'d inside its trace so it survives).
            scores = detect_fn(model, model_id, lora_id, batch, **detect_kwargs)
            pieces.append(scores)
            poses.append(torch.tensor(bpos))

        # Concatenate + .save() ONCE, at the very end (still inside the session).
        # Only saved objects survive the `with`, so we read THESE after it — never
        # the plain loop locals (toks/order/pieces), which don't.
        all_scores = torch.cat(pieces).save()
        all_pos = torch.cat(poses).save()

    flat = all_scores.cpu().float().numpy()
    pos = all_pos.cpu().numpy()
    result = np.empty(len(flat), dtype=float)
    result[pos] = flat                                   # back to original row order
    return result


# ── output ───────────────────────────────────────────────────────────────────

def write_submission(indices, scores, path: str = "submission.csv", *,
                     threshold: float = 0.5, deceptive=None) -> str:
    """Write the grader's three columns. ``score`` is clipped to ``[0, 1]``;
    ``deceptive`` defaults to ``score >= threshold`` (pass your own boolean array to
    override). ``indices`` and ``scores`` must be the same length and aligned."""
    import numpy as np
    import pandas as pd

    scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    indices = list(indices)
    if len(indices) != len(scores):
        raise ValueError(f"indices ({len(indices)}) and scores ({len(scores)}) differ in length")
    if deceptive is None:
        deceptive = scores >= threshold
    pd.DataFrame({"index": indices, "deceptive": np.asarray(deceptive, dtype=bool),
                  "score": scores}).to_csv(path, index=False)
    print(f"wrote {len(scores)} rows to {path}")
    return path
