from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]


def compose_config(name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs").resolve()),
    ):
        return compose(config_name=name)


def test_pid_persona_student_is_a_matched_varied_only_ablation():
    control = compose_config("privileged_information_distillation")
    persona = compose_config(
        "privileged_information_distillation_persona_epistemic_variedonly"
    )

    assert persona.student.model == control.student.model
    assert persona.student.lora == control.student.lora
    assert persona.student.target_mode == control.student.target_mode == "teacher"
    assert persona.student.dataset_name_contains == "varied-deception"
    assert persona.student.override_cached_prompt is False
    assert persona.student.training.learning_rate == 5e-5
    assert persona.student.training.num_train_epochs == 1.0
    assert persona.teacher.dataset_name_contains == "varied-deception"
    assert persona.method in persona.teacher.artifact
    assert not persona.teacher.artifact.endswith(
        "qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl"
    )


def test_pid_persona_prompt_preserves_contract_and_narrow_guards():
    persona = compose_config(
        "privileged_information_distillation_persona_epistemic_variedonly"
    )
    prompt = persona.student.prompt.lower()
    teacher_prompt = persona.teacher.prompt.lower()

    assert "underlying assistant" in prompt
    assert "definite knowledge reports" in prompt
    assert "policy refusal" in prompt
    assert "genuinely obscure details" in prompt
    assert "alone is insufficient" in prompt
    assert "<reasoning_summary>" in prompt
    assert "prediction:<0 or 1>" in prompt
    assert "underlying assistant" in teacher_prompt
    assert "accept this label as correct" in teacher_prompt
    assert "prediction:__ground_truth__" in teacher_prompt


def test_pid_matchedrows_reuses_new_teacher_and_fixes_control_keys():
    matched = compose_config(
        "privileged_information_distillation_persona_epistemic_matchedrows"
    )

    assert matched.method == (
        "qwen9b_pid_persona_epistemic_matchedrows_adamw5e5_v1"
    )
    assert matched.teacher.artifact.endswith(
        "qwen9b_pid_persona_epistemic_variedonly_adamw5e5_v1/"
        "teacher/train.jsonl"
    )
    assert matched.student.selection_manifest.endswith(
        "qwen9b_pid_persona_epistemic_variedonly_adamw5e5_v1/"
        "teacher/control_usable_manifest.jsonl"
    )
    assert matched.student.dataset_name_contains == "varied-deception"
    assert matched.student.training.learning_rate == 5e-5
