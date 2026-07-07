"""Text normalization helpers for Phoenix Wright prompt construction."""

from __future__ import annotations


def safe_text(value) -> str:
    """Return text that can be passed to Rust-backed Hugging Face tokenizers."""
    return str(value).encode("utf-8", "replace").decode("utf-8")
