from legacy_submissions.phoenix_wright_v2_scoring import reply_to_score


def test_reply_to_score_parses_binary_prediction() -> None:
    assert reply_to_score("<reasoning_summary>Evidence.</reasoning_summary>\nPrediction:1") == 1.0


def test_reply_to_score_uses_last_explicit_prediction() -> None:
    assert reply_to_score("Prediction:1\nCorrection: Prediction:0") == 0.0


def test_reply_to_score_defaults_unparseable_output_to_zero() -> None:
    assert reply_to_score("The answer is probably deceptive.") == 0.0
