from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phoenix_renderer_caps.run import (
    CONDITIONS,
    build_member_prompt,
    metric_report,
    position_batches,
)
from legacy_submissions.phoenix_wright_v3 import (
    build_member_prompt as historical_member_prompt,
)


def test_current_condition_is_byte_exact_phoenix4_renderer() -> None:
    messages = [
        {"role": "system", "content": "s" * 2_100},
        {"role": "user", "content": "u" * 1_700},
        {"role": "assistant", "content": "a" * 3_500},
    ]
    prompt, audit = build_member_prompt(messages, CONDITIONS[0])

    assert prompt == historical_member_prompt(messages)
    assert audit["message_cap_fired"]
    assert audit["context_cap_fired"]
    assert audit["output_cap_fired"]


def test_larger_caps_preserve_more_long_message_text() -> None:
    messages = [
        {"role": "user", "content": "u" * 2_500},
        {"role": "assistant", "content": "a" * 5_000},
    ]
    current, _ = build_member_prompt(messages, CONDITIONS[0])
    doubled, _ = build_member_prompt(messages, CONDITIONS[1])
    quadrupled, _ = build_member_prompt(messages, CONDITIONS[3])

    assert len(current) < len(doubled) < len(quadrupled)


def test_dynamic_batches_add_safe_long_prompt_token_budget() -> None:
    lengths = [500] * 48 + [700] * 32 + [1_000] * 16
    assert [len(batch) for batch in position_batches(lengths)] == [48, 32, 16]

    very_long = [2_000] * 16 + [4_000] * 8 + [8_000] * 4
    batches = position_batches(very_long)
    for batch in batches:
        longest = max(very_long[position] for position in batch)
        assert len(batch) * longest <= 32_768


def test_metric_report_supports_single_long_context_corpus() -> None:
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({
        "dataset": ["insider"] * 4,
        "scenario": ["insider-trading"] * 4,
        "source_model": ["a", "a", "b", "b"],
        "label": [0, 1, 0, 1],
    })
    report = metric_report(frame, np.array([0.1, 0.9, 0.2, 0.8]))

    assert report["auroc"] == 1.0
    assert report["macro_auroc"] == 1.0
    assert report["per_source_model_auroc"] == {"a": 1.0, "b": 1.0}
    assert "scenario_auroc" not in report
