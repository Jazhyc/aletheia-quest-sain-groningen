"""Parsing for the Phoenix Wright v2 distilled binary judge."""

from __future__ import annotations

import re


_PREDICTION_RE = re.compile(r"(?im)\bPrediction\s*:\s*([01])\b")


def reply_to_score(reply: object) -> float:
    """Return the final explicit binary prediction, defaulting safely to zero."""
    matches = _PREDICTION_RE.findall(str(reply if reply is not None else ""))
    return float(matches[-1]) if matches else 0.0
