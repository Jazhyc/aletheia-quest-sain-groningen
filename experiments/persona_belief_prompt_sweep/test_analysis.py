import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.persona_belief_prompt_sweep.analyze import (
    is_persona_roleplay,
    safe_auroc,
    summarize_member,
)


def test_persona_roleplay_detection_uses_system_text_only():
    messages = [
        {
            "role": "system",
            "content": "You are playing the character of a Roman citizen.",
        },
        {"role": "user", "content": "No roleplay wording here."},
        {"role": "assistant", "content": "I do not know that country."},
    ]
    assert is_persona_roleplay(messages)
    assert not is_persona_roleplay([
        {"role": "system", "content": "Answer ordinary factual questions."},
        {"role": "assistant", "content": "Let's play a game."},
    ])


def test_safe_auroc_requires_both_classes():
    assert safe_auroc(np.asarray([0, 0]), np.asarray([0.1, 0.2])) is None
    assert safe_auroc(np.asarray([0, 1]), np.asarray([0.1, 0.9])) == 1.0


def test_summary_selects_macro_dataset_auroc_and_reports_continuity():
    frames = [
        pd.DataFrame({
            "dataset": ["a", "a", "a", "a"],
            "label": [0, 0, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.9],
            "persona": [True, False, True, False],
        }),
        pd.DataFrame({
            "dataset": ["b", "b", "b", "b"],
            "label": [0, 0, 1, 1],
            "score": [0.9, 0.8, 0.2, 0.1],
            "persona": [True, False, True, False],
        }),
    ]
    summary = summarize_member(frames)
    assert summary["macro_per_dataset_auroc"] == 0.5
    assert summary["score_continuity"] == {
        "n": 8,
        "unique_scores": 4,
        "ties": 4,
        "unique_fraction": 0.5,
    }
