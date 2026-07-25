"""Continuous direct-label scoring helpers for Phoenix Wright 4.0."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from phoenix_wright_v3 import MEMBER_NAMES, build_member_prompt
except ImportError:  # package import in repository tests
    from .phoenix_wright_v3 import MEMBER_NAMES, build_member_prompt


DIRECT_PREDICTION_PREFIX = "Prediction:"


def build_direct_prompt(messages: Any, tokenizer: Any) -> str:
    """Render the frozen judge prompt at its direct binary decision position."""
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_member_prompt(messages)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return chat + DIRECT_PREDICTION_PREFIX


def binary_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Return the distinct single-token ids for the exact labels ``0`` and ``1``."""
    ids = []
    for label in ("0", "1"):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"binary label {label!r} tokenized as {encoded}; expected one token"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != 2:
        raise ValueError(f"binary labels must have distinct token ids, got {ids}")
    return ids[0], ids[1]


def binary_logits_to_scores(logits: Any) -> np.ndarray:
    """Normalize ``[logit_0, logit_1]`` rows into ``P(Prediction=1)``."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected logits shaped (rows, 2), got {values.shape}")
    difference = np.clip(values[:, 1] - values[:, 0], -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-difference))
