import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qwen_grpo_lora import SplitRecords, filter_datasets


def records() -> SplitRecords:
    return SplitRecords(
        frame=pd.DataFrame({"dataset": ["dev-varied-deception-a", "dev-instructed-deception-b"]}),
        dataset_names=["dev-varied-deception-a", "dev-instructed-deception-b"],
    )


def test_filter_datasets_uses_literal_substring() -> None:
    filtered = filter_datasets(records(), "varied-deception")
    assert filtered.frame["dataset"].tolist() == ["dev-varied-deception-a"]
    assert filtered.dataset_names == ["dev-varied-deception-a"]


def test_filter_datasets_rejects_empty_selection() -> None:
    with pytest.raises(RuntimeError, match="no datasets"):
        filter_datasets(records(), "missing")
