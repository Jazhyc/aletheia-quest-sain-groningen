import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.evaluate_student_sft import (
    ATOMIC_AUDIT_STRICT_RE,
    COUNTERFACTUAL_STRICT_RE,
    PREDICTION_ONLY_STRICT_RE,
    RATING_STRICT_RE,
    apply_student_prompt_config,
    binary_score_from_logprobs,
    binary_token_ids,
    comparable_student_settings,
    load_retrieval_cache,
    max_aggregate_evaluations,
    parse_reasoning_input_condition,
    parse_prompt_condition,
    parse_rating,
    parse_retrieval_condition,
    parse_prediction,
    prefix_before_prediction,
    rating_to_score,
    set_reasoning_visibility,
    strict_pattern_for_config,
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


def test_parse_rating_uses_last_explicit_rating() -> None:
    assert parse_rating("Rating:2\nCorrection: Rating:<6>") == 6


def test_parse_rating_rejects_binary_prediction_and_out_of_range_value() -> None:
    assert parse_rating("Prediction:1") is None
    assert parse_rating("Rating:8") is None


def test_rating_to_score_uses_evenly_spaced_endpoints_and_midpoint() -> None:
    assert rating_to_score(1) == 0.0
    assert rating_to_score(4) == 0.5
    assert rating_to_score(7) == 1.0


def test_apply_student_prompt_config_overrides_conditional_prompt(tmp_path) -> None:
    path = tmp_path / "prompt.yaml"
    path.write_text(
        "student:\n"
        "  prompt: trace prompt\n"
        "  prompt_without_reasoning: ordinary prompt\n"
        "  exclude_final_output_from_context: true\n"
        "  target_mode: prediction_only\n"
    )
    config = {"student": {"prompt": "old", "prompt_without_reasoning": "old fallback"}}

    apply_student_prompt_config(config, path)

    assert config["student"]["prompt"] == "trace prompt"
    assert config["student"]["prompt_without_reasoning"] == "ordinary prompt"
    assert config["student"]["exclude_final_output_from_context"] is True
    assert config["student"]["target_mode"] == "prediction_only"
    assert strict_pattern_for_config(config) is PREDICTION_ONLY_STRICT_RE


def test_max_aggregate_evaluations_aligns_rows_and_preserves_members() -> None:
    first = pd.DataFrame([
        {
            "dataset": "d",
            "index": 1,
            "label": 1,
            "score": 0.0,
            "parse_error": False,
            "format_valid": True,
            "generation": "Prediction:0",
            "prompt_sha256": "a1",
        },
        {
            "dataset": "d",
            "index": 2,
            "label": 0,
            "score": 0.0,
            "parse_error": True,
            "format_valid": False,
            "generation": "",
            "prompt_sha256": "a2",
        },
    ])
    second = pd.DataFrame([
        {
            "dataset": "d",
            "index": 2,
            "label": 0,
            "score": 0.0,
            "parse_error": False,
            "format_valid": True,
            "generation": "Prediction:0",
            "prompt_sha256": "b2",
        },
        {
            "dataset": "d",
            "index": 1,
            "label": 1,
            "score": 1.0,
            "parse_error": False,
            "format_valid": True,
            "generation": "Prediction:1",
            "prompt_sha256": "b1",
        },
    ])

    result = max_aggregate_evaluations({"summary": first, "binary": second})

    assert result["score"].tolist() == [1.0, 0.0]
    assert result["prediction"].tolist() == [1.0, 0.0]
    assert result["parse_error"].tolist() == [False, False]
    assert result["summary_score"].tolist() == [0.0, 0.0]
    assert result["binary_score"].tolist() == [1.0, 0.0]
    assert json.loads(result.loc[0, "generation"]) == {
        "summary": "Prediction:0",
        "binary": "Prediction:1",
    }


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


def test_comparable_student_settings_distinguishes_reasoning_visibility() -> None:
    common = {
        "model": "model",
        "prompt": "prompt",
        "max_prompt_chars": 3000,
        "context_truncation": "tail",
    }
    without_reasoning = comparable_student_settings({"student": common})
    with_reasoning = comparable_student_settings({
        "student": {
            **common,
            "include_reasoning": True,
            "reasoning_max_chars": 2400,
            "reasoning_truncation": "head_tail",
        }
    })

    assert without_reasoning != with_reasoning


def test_set_reasoning_visibility_can_hide_configured_trace() -> None:
    config = {"student": {"include_reasoning": True, "reasoning_max_chars": 2400}}

    set_reasoning_visibility(config, "hidden")

    assert config["student"]["include_reasoning"] is False
    assert config["student"]["reasoning_max_chars"] == 2400


def test_set_reasoning_visibility_preserves_configured_trace() -> None:
    config = {"student": {"include_reasoning": True}}

    set_reasoning_visibility(config, "configured")

    assert config["student"]["include_reasoning"] is True


def test_parse_retrieval_condition_supports_empty_and_named_fields(tmp_path) -> None:
    assert parse_retrieval_condition("empty", tmp_path) == (
        "empty", None, "passages"
    )
    assert parse_retrieval_condition(
        "shuffled=cache.jsonl#shuffled_passages", tmp_path
    ) == (
        "shuffled",
        (tmp_path / "cache.jsonl").resolve(),
        "shuffled_passages",
    )


def test_parse_reasoning_input_condition() -> None:
    assert parse_reasoning_input_condition("long=4000") == (
        "long",
        4000,
        "head_tail",
    )
    assert parse_reasoning_input_condition("head=1200:head") == (
        "head",
        1200,
        "head",
    )
    assert parse_reasoning_input_condition("tail_heavy=4000:head_tail_25") == (
        "tail_heavy",
        4000,
        "head_tail_25",
    )


def test_parse_prompt_condition(tmp_path) -> None:
    assert parse_prompt_condition("strict=config.yaml", tmp_path) == (
        "strict",
        (tmp_path / "config.yaml").resolve(),
    )


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


def test_rating_strict_format_requires_summary_and_one_to_seven_rating() -> None:
    valid = "<reasoning_summary>Evidence.</reasoning_summary>\nRating:5"

    assert RATING_STRICT_RE.fullmatch(valid)
    assert not RATING_STRICT_RE.fullmatch(valid.replace("Rating:5", "Rating:8"))
    assert strict_pattern_for_config(
        {"student": {"target_format": "rating"}}
    ) is RATING_STRICT_RE


def test_atomic_audit_strict_format_accepts_grounded_audit_contract() -> None:
    valid = (
        "<factual_claims>\n<claim>\n"
        "<quote>Paris is in France</quote>\n"
        "<proposition>Paris is in France.</proposition>\n"
        "<assessment>true</assessment>\n</claim>\n</factual_claims>\n"
        "<context_claims></context_claims>\n"
        "<audit_label_relation>supports</audit_label_relation>\n"
        "<reasoning_summary>The claim is accurate.</reasoning_summary>\n"
        "Prediction:0"
    )

    assert ATOMIC_AUDIT_STRICT_RE.fullmatch(valid)
    assert not ATOMIC_AUDIT_STRICT_RE.fullmatch(
        valid.replace("<assessment>true</assessment>\n", "")
    )
    assert strict_pattern_for_config(
        {"student": {"target_format": "atomic_audit"}}
    ) is ATOMIC_AUDIT_STRICT_RE


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
