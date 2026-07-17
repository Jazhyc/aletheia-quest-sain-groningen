from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "details": "pid_student_dks_details_v1.yaml",
    "known": "pid_student_dks_known_v1.yaml",
    "scrutiny": "pid_student_dks_scrutiny_v1.yaml",
}
RATING_CONFIGS = {
    "details": "pid_student_dks_details_rating7_v1.yaml",
    "known": "pid_student_dks_known_rating7_v1.yaml",
    "scrutiny": "pid_student_dks_scrutiny_rating7_v1.yaml",
}


def load_prompt(name: str) -> str:
    config = yaml.safe_load((ROOT / "configs" / CONFIGS[name]).read_text())
    assert config["student"]["include_reasoning"] is False
    return config["student"]["prompt"]


def load_rating_prompt(name: str) -> str:
    config = yaml.safe_load((ROOT / "configs" / RATING_CONFIGS[name]).read_text())
    assert config["student"]["include_reasoning"] is False
    assert config["student"]["target_format"] == "rating"
    return config["student"]["prompt"]


def test_pid_dks_prompts_preserve_binary_summary_contract() -> None:
    for name in CONFIGS:
        prompt = load_prompt(name)
        assert "<reasoning_summary>" in prompt
        assert "</reasoning_summary>" in prompt
        assert prompt.endswith("Prediction:<0 or 1>")
        assert "1 means deceptive and 0 means not deceptive" in prompt


def test_pid_dks_prompts_have_distinct_decision_frames() -> None:
    assert "direct answer" in load_prompt("details")
    assert "context states the true answer" in load_prompt("known")
    assert "Check briefly:" in load_prompt("scrutiny")


def test_pid_dks_rating_prompts_preserve_summary_and_ordinal_contract() -> None:
    for name in RATING_CONFIGS:
        prompt = load_rating_prompt(name)
        assert "<reasoning_summary>" in prompt
        assert "</reasoning_summary>" in prompt
        assert prompt.endswith("Rating:<integer from 1 to 7>")
        assert "1 means definitely not deceptive" in prompt
        assert "7 means definitely deceptive" in prompt
        assert "Prediction:" not in prompt


def test_pid_dks_rating_prompts_keep_distinct_decision_frames() -> None:
    assert "direct answer" in load_rating_prompt("details")
    assert "context states the true answer" in load_rating_prompt("known")
    assert "Check briefly:" in load_rating_prompt("scrutiny")
