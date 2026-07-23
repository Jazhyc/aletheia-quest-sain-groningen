from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reasoning_ngram_intent.train import (
    Candidate,
    balanced_dataset_label_weights,
    candidate_grid,
    make_vectorizer,
    reasoning_view,
)
from experiments.privileged_information_distillation.core import (
    final_assistant_reasoning,
)


def test_extracts_only_final_assistant_reasoning() -> None:
    messages = [
        {"role": "assistant", "content": "draft", "reasoning": "old plan"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "answer", "reasoning": "final plan"},
    ]

    assert final_assistant_reasoning(messages) == "final plan"


def test_reasoning_view_never_includes_output_text() -> None:
    candidate = Candidate(
        truncation="tail",
        max_chars=8,
        min_df=3,
        c=0.1,
    )

    assert reasoning_view("REASONING_ONLY", candidate) == "ING_ONLY"


def test_balanced_weights_equalize_dataset_label_cells() -> None:
    frame = pd.DataFrame({
        "dataset": ["a", "a", "a", "b", "b", "b"],
        "label": [0, 0, 1, 0, 1, 1],
    })

    weights = balanced_dataset_label_weights(frame)
    totals = {}
    for cell, weight in zip(
        zip(frame["dataset"], frame["label"], strict=True),
        weights,
        strict=True,
    ):
        totals[cell] = totals.get(cell, 0.0) + weight

    assert np.allclose(list(totals.values()), list(totals.values())[0])


def test_grid_and_vectorizer_keep_capacity_bounded() -> None:
    grid = candidate_grid()

    assert len(grid) == 16
    assert {candidate.truncation for candidate in grid} == {"tail", "head_tail"}
    assert {candidate.c for candidate in grid} == {0.03, 0.1, 0.3, 1.0}
    vectorizer = make_vectorizer(grid[0])
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.max_features == 20_000
