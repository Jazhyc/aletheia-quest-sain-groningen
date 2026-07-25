import json

from experiments.privileged_information_distillation.audit_teacher_rationales import (
    audit_prompt,
    heuristic_flags,
    parse_json_object,
    record_key,
    recommended_use,
    retry_keys,
    summarize,
    validate_audit,
)


def valid_audit(**overrides):
    value = {
        "implied_prediction": 1,
        "factual_status": "sound",
        "logic_status": "coherent",
        "meta_status": "none",
        "problem_quote": "NONE",
        "explanation": "The summary supports a deceptive verdict.",
    }
    value.update(overrides)
    return value


def test_audit_prompt_is_label_blind():
    record = {
        "label": 1,
        "student_prompt": (
            "Judge final <output> under the rule.\n"
            "<context>\nUSER: Question\nASSISTANT: Answer\n</context>\n"
            "<output>\nAnswer\n</output>"
        ),
        "reasoning_summary": "The answer makes an important false claim.",
    }
    prompt = audit_prompt(record)
    assert "authoritative binary training label is deliberately\nnot shown" in prompt
    assert "<assistant_output>\nAnswer\n</assistant_output>" in prompt
    assert "<teacher_summary>" in prompt
    assert "Do not use \"confident_error\" merely because the output is false." in prompt
    assert "Summary says the output contains an important false claim" in prompt
    assert "Prediction:1" not in prompt


def test_parse_and_validate_harmony_json():
    completion = (
        "analysis channel text\n<|channel|>final<|message|>"
        '{"implied_prediction":0,"factual_status":"sound",'
        '"logic_status":"coherent","meta_status":"none",'
        '"problem_quote":"NONE","explanation":"Coherent."}'
    )
    parsed = parse_json_object(completion)
    audit, errors = validate_audit(parsed)
    assert errors == []
    assert audit is not None
    assert audit["implied_prediction"] == 0


def test_validate_normalizes_string_prediction():
    audit, errors = validate_audit(valid_audit(implied_prediction="1"))
    assert errors == []
    assert audit is not None
    assert audit["implied_prediction"] == 1


def test_record_key_normalizes_index():
    assert record_key({"dataset": "unit", "index": "12"}) == ("unit", 12)


def test_retry_keys_reparses_prior_raw_completions(tmp_path):
    path = tmp_path / "audit.jsonl"
    valid = valid_audit(implied_prediction=0)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "unit",
                        "index": 1,
                        "raw_completion": json.dumps(valid),
                    }
                ),
                json.dumps(
                    {
                        "dataset": "unit",
                        "index": 2,
                        "raw_completion": "truncated before JSON",
                    }
                ),
            ]
        )
        + "\n"
    )
    assert retry_keys(path) == {("unit", 2)}


def test_recommended_use_requires_clean_matching_summary():
    clean = {"label": 1, "audit": valid_audit(), "audit_errors": []}
    conflict = {
        "label": 1,
        "audit": valid_audit(implied_prediction=0),
        "audit_errors": [],
    }
    leakage = {
        "label": 1,
        "audit": valid_audit(meta_status="generic_label_reference"),
        "audit_errors": [],
    }
    explicit_leakage = {
        "label": 1,
        "audit": valid_audit(meta_status="explicit_privileged_reference"),
        "audit_errors": [],
    }
    unclear = {
        "label": 1,
        "audit": valid_audit(implied_prediction="unclear"),
        "audit_errors": [],
    }
    malformed = {"label": 1, "audit": None, "audit_errors": ["json_parse"]}
    assert recommended_use(clean) == "keep_summary"
    assert recommended_use(conflict) == "label_only"
    assert recommended_use(leakage) == "manual_review"
    assert recommended_use(explicit_leakage) == "label_only"
    assert recommended_use(unclear) == "manual_review"
    assert recommended_use(malformed) == "manual_review"


def test_heuristic_meta_flags_do_not_match_ordinary_labeling_verbs():
    ordinary = {
        "label": 1,
        "reasoning_summary": "The assistant labels the false statement as true.",
    }
    leaked = {
        "label": 1,
        "reasoning_summary": (
            "The answer is true, making the label unsupported under the "
            "privileged panel instruction."
        ),
    }
    assert heuristic_flags(ordinary) == []
    assert set(heuristic_flags(leaked)) == {
        "positive_truth_conclusion",
        "label_reference",
        "privileged_reference",
    }


def test_summarize_stratifies_failures():
    records = [
        {
            "dataset": "dev-instructed-deception-model",
            "label": 0,
            "audit": valid_audit(implied_prediction=0),
            "audit_errors": [],
            "heuristic_flags": [],
        },
        {
            "dataset": "dev-varied-deception-model",
            "label": 1,
            "audit": valid_audit(
                implied_prediction=0,
                logic_status="contradicts_own_conclusion",
            ),
            "audit_errors": [],
            "heuristic_flags": ["positive_truth_conclusion"],
        },
        {
            "dataset": "dev-varied-deception-model",
            "label": 1,
            "audit": valid_audit(implied_prediction="unclear"),
            "audit_errors": [],
            "heuristic_flags": [],
        },
    ]
    for record in records:
        record["recommended_use"] = recommended_use(record)
    report = summarize(records)
    assert report["counts"]["records"] == 3
    assert report["counts"]["implied_label_conflict"] == 1
    assert report["counts"]["label_only"] == 1
    assert report["counts"]["implied_label_unclear"] == 1
    assert report["counts"]["manual_review"] == 1
    assert report["by_scenario"]["varied"]["implied_label_conflict"] == 1
    assert report["by_label"]["1"]["logic_contradicts_own_conclusion"] == 1
