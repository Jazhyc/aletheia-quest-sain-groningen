from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (
    "pid_heterogeneous_belief_statement_rank1_v1.yaml",
    "pid_heterogeneous_belief_proof_rank1_v1.yaml",
)


def load_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_intent_prompts_use_blind_belief_statement_definition() -> None:
    for name in CONFIGS:
        config = load_config(name)
        teacher = config["teacher"]["prompt"]
        student = config["student"]["prompt"]

        assert "__GROUND_TRUTH__" not in teacher
        assert "own factual knowledge" in teacher or "own fact checking" in teacher
        assert "final <output>" in teacher
        assert "believe" in teacher
        assert teacher.endswith("Prediction:<0 or 1>")
        assert student.endswith("Prediction:<0 or 1>")


def test_intent_students_keep_trace_conditioning_and_rank1_manifest() -> None:
    for name in CONFIGS:
        student = load_config(name)["student"]

        assert student["include_reasoning"] is True
        assert student["reasoning_max_chars"] == 4000
        assert student["reasoning_truncation"] == "tail"
        assert student["selection_manifest"].endswith(
            "resolved_intent_parsed_train.jsonl"
        )
