from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.liars_bench_signature.evaluate_signatures import parse_score, stratified_sample


def test_parse_rating_and_binary_conditions() -> None:
    assert parse_score("Rating: 7", "truth") == (1.0, False)
    assert parse_score("Rating: 1", "details") == (0.0, False)
    assert parse_score("Prediction:1", "phoenix_v21_adapter") == (1.0, False)
    assert parse_score("unknown", "omission") == (0.0, True)


def test_stratified_sample_keeps_labels_and_limit() -> None:
    frame = pd.DataFrame({
        "sample_id": [str(i) for i in range(40)],
        "label": [0] * 20 + [1] * 20,
        "source_model": ["a"] * 10 + ["b"] * 10 + ["a"] * 10 + ["b"] * 10,
    })
    sampled = stratified_sample(frame, 12, 7)
    assert len(sampled) == 12
    assert sampled.groupby(["label", "source_model"]).size().to_dict() == {
        (0, "a"): 3, (0, "b"): 3, (1, "a"): 3, (1, "b"): 3
    }
