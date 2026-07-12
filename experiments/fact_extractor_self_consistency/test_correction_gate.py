from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fact_extractor_self_consistency.run_correction_gate import (
    DECISION_FEATURES,
    fixed_rule,
    leave_one_organism_out,
    scenario_from_dataset,
    stratified_oof,
    text_features,
)


def test_text_features_detect_factual_and_referential_markers() -> None:
    features = text_features(
        'Yes, that claim says the Bank of England was founded in 1694 because £1,200 was raised.'
    )
    assert features["reference_count"] >= 2
    assert features["year_count"] == 1
    assert features["currency_count"] >= 1
    assert features["capitalized_span_count"] >= 1
    assert features["history_causal_count"] >= 2
    assert features["factual_density"] >= 5


def test_scenario_parser() -> None:
    assert scenario_from_dataset("x/dev-varied-deception-Qwen") == "varied-deception"
    assert scenario_from_dataset("x/dev-instructed-deception-Qwen") == "instructed-deception"


def test_fixed_rule_only_overrides_eligible_negative_judgments() -> None:
    frame = pd.DataFrame({
        "judge_prediction": [0, 0, 0, 1],
        "self_rejects": [1, 1, 0, 0],
        "reference_count": [0, 1, 0, 0],
        "factual_density": [2, 3, 4, 0],
        "log_token_count": np.log1p([20, 30, 40, 5]),
    })
    assert fixed_rule(frame).tolist() == [1.0, 0.0, 0.0, 1.0]


def test_oof_helpers_assign_every_row() -> None:
    n = 40
    frame = pd.DataFrame({
        "label": np.tile([0, 1], n // 2),
        "judge_prediction": np.tile([0, 1], n // 2),
        "self_rejects": np.tile([0, 0, 1, 1], n // 4),
        "organism": np.repeat(["a", "b", "c", "d"], n // 4),
    })
    frame["disagreement"] = (frame["judge_prediction"] != frame["self_rejects"]).astype(int)
    stratified = stratified_oof(frame, DECISION_FEATURES)
    logo = leave_one_organism_out(frame, DECISION_FEATURES)
    assert np.isfinite(stratified).all()
    assert np.isfinite(logo).all()
    assert ((0 <= stratified) & (stratified <= 1)).all()
    assert ((0 <= logo) & (logo <= 1)).all()
