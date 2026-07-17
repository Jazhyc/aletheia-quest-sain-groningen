from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "details": "pid_student_dks_details_v1.yaml",
    "known": "pid_student_dks_known_v1.yaml",
    "scrutiny": "pid_student_dks_scrutiny_v1.yaml",
}


def load_prompt(name: str) -> str:
    config = yaml.safe_load((ROOT / "configs" / CONFIGS[name]).read_text())
    assert config["student"]["include_reasoning"] is False
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
