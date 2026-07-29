from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_backend_parity.run import (
    binary_token_ids,
    position_batches,
    score_from_logprobs,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {"0": [10], "1": [11]}[text]


def test_binary_score_normalizes_requested_vllm_logprobs() -> None:
    values = {
        10: SimpleNamespace(logprob=-2.0),
        11: SimpleNamespace(logprob=-1.0),
    }
    assert binary_token_ids(_Tokenizer()) == [10, 11]
    assert score_from_logprobs(values, [10, 11]) == pytest.approx(
        0.7310585786
    )


def test_position_batches_apply_phoenix_4_tiers() -> None:
    lengths = [500] * 48 + [700] * 32 + [1000] * 16
    batches = position_batches(lengths)
    assert [len(batch) for batch in batches] == [48, 32, 16]
