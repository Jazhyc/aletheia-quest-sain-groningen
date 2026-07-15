import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.evaluate_student_sft import (
    COUNTERFACTUAL_STRICT_RE,
    PREDICTION_ONLY_STRICT_RE,
    binary_score_from_logprobs,
    binary_token_ids,
    comparable_student_settings,
    load_retrieval_cache,
    parse_prediction,
    prefix_before_prediction,
)
from experiments.privileged_information_distillation.analyze_continuous_margins import (
    rank_fraction_predictions,
)


def test_parse_prediction_uses_last_explicit_prediction() -> None:
    text = "<reasoning_summary>Evidence.</reasoning_summary>\nPrediction:0\nPrediction: 1"
    assert parse_prediction(text) == 1


def test_parse_prediction_rejects_unstructured_binary_text() -> None:
    assert parse_prediction("The answer is 1.") is None


def test_parse_prediction_accepts_literal_placeholder_brackets() -> None:
    assert parse_prediction("Prediction:<0>") == 0


def test_load_retrieval_cache_formats_passages(tmp_path) -> None:
    path = tmp_path / "rag.jsonl"
    path.write_text(json.dumps({
        "dataset": "dataset",
        "index": 7,
        "passages": [{"title": "Cerberus", "text": "Three-headed guardian."}],
    }) + "\n")

    assert load_retrieval_cache(path) == {
        ("dataset", 7): "- Cerberus: Three-headed guardian."
    }


def test_load_retrieval_cache_selects_named_passage_field(tmp_path) -> None:
    path = tmp_path / "rag.jsonl"
    path.write_text(json.dumps({
        "dataset": "dataset",
        "index": 7,
        "real_passages": [{"title": "Real", "text": "Relevant."}],
        "shuffled_passages": [{"title": "Donor", "text": "Irrelevant."}],
    }) + "\n")

    assert load_retrieval_cache(path, passage_field="shuffled_passages") == {
        ("dataset", 7): "- Donor: Irrelevant."
    }


def test_comparable_student_settings_normalizes_legacy_target_defaults() -> None:
    common = {
        "model": "model",
        "prompt": "prompt",
        "max_prompt_chars": 3000,
        "context_truncation": "tail",
    }

    assert comparable_student_settings({"student": common}) == comparable_student_settings({
        "student": {**common, "target_format": "summary", "target_mode": "teacher"}
    })


def test_counterfactual_strict_format_requires_ordered_fields() -> None:
    valid = (
        "<reasoning_summary>Summary.</reasoning_summary>\n"
        "<facts>Facts.</facts>\n"
        "<contradiction>No contradiction.</contradiction>\nPrediction:0"
    )

    assert COUNTERFACTUAL_STRICT_RE.fullmatch(valid)
    assert not COUNTERFACTUAL_STRICT_RE.fullmatch(
        valid.replace("<facts>Facts.</facts>\n", "")
    )


def test_prediction_only_strict_format_accepts_binary_contract() -> None:
    assert PREDICTION_ONLY_STRICT_RE.fullmatch("Prediction:1")
    assert PREDICTION_ONLY_STRICT_RE.fullmatch("Prediction:<0>")
    assert not PREDICTION_ONLY_STRICT_RE.fullmatch("Reasoning\nPrediction:1")


def test_prefix_before_prediction_removes_only_final_prediction_token() -> None:
    text = "<reasoning_summary>Prediction:0 is quoted.</reasoning_summary>\nPrediction:1"
    assert prefix_before_prediction(text) == text[:-1]


def test_prefix_before_prediction_adds_field_after_unparsed_generation() -> None:
    assert prefix_before_prediction("unfinished reasoning ") == (
        "unfinished reasoning\nPrediction:"
    )


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return {"0": [10], "1": [11]}[text]


def test_binary_token_ids_require_distinct_single_tokens() -> None:
    assert binary_token_ids(_Tokenizer()) == [10, 11]


class _Logprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


def test_binary_score_normalizes_zero_one_logprob_margin() -> None:
    score = binary_score_from_logprobs(
        {10: _Logprob(-2.0), 11: _Logprob(-1.0)},
        [10, 11],
    )
    assert score is not None
    assert abs(score - 0.7310585786) < 1e-9


def test_binary_score_rejects_missing_target_logprob() -> None:
    assert binary_score_from_logprobs({10: _Logprob(-1.0)}, [10, 11]) is None


def test_rank_fraction_predictions_operates_per_dataset() -> None:
    frame = pd.DataFrame({
        "dataset": ["a", "a", "b", "b"],
        "index": [2, 1, 3, 4],
        "label": [0, 1, 0, 1],
        "margin": [0.8, 0.8, 0.1, 0.9],
    })
    ranked = rank_fraction_predictions(frame, "margin", 0.5)
    assert ranked.groupby("dataset")["score"].sum().to_dict() == {"a": 1.0, "b": 1.0}
    assert ranked.loc[ranked["score"] == 1.0, "index"].tolist() == [1, 4]
