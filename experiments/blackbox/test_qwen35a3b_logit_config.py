from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.blackbox.run_judge import (
    missing_requested_token_ids,
    normalize_rating_probs,
    strip_terminal_generated_rating,
    vllm_kwargs_from_config,
)


ROOT = Path(__file__).resolve().parents[2]


def compose_ensemble(name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs" / "judge_ensemble").resolve()),
    ):
        return compose(config_name=name)


def test_qwen35a3b_teacher_gate_is_an_exact_model_swap() -> None:
    control = compose_ensemble(
        "blackbox_reasoning_ensemble_dks3072_logit_v1"
    )
    candidate = compose_ensemble(
        "blackbox_reasoning_ensemble_dks3072_logit_qwen35a3b_v1"
    )

    assert candidate.judge.model == "Qwen/Qwen3.5-35B-A3B"
    assert candidate.judge.mode == control.judge.mode == "logits"
    assert (
        OmegaConf.select(candidate, "judge.logit_prefix", default="Rating:")
        == OmegaConf.select(control, "judge.logit_prefix", default="Rating:")
        == "Rating:"
    )
    assert candidate.judge.rating_min == control.judge.rating_min == 1
    assert candidate.judge.rating_max == control.judge.rating_max == 7
    assert candidate.judge.max_prompt_chars == control.judge.max_prompt_chars
    assert candidate.ensemble.aggregation == control.ensemble.aggregation == "max"
    assert candidate.ensemble.order == control.ensemble.order == "member"
    assert candidate.ensemble.members == control.ensemble.members
    assert candidate.scoring.threshold == control.scoring.threshold


def test_qwen35a3b_teacher_gate_fits_one_rtx_pro_6000() -> None:
    candidate = compose_ensemble(
        "blackbox_reasoning_ensemble_dks3072_logit_qwen35a3b_v1"
    )

    assert candidate.judge.dtype == "bfloat16"
    assert candidate.judge.tensor_parallel_size == 1
    assert candidate.judge.max_model_len == 4096
    assert candidate.judge.max_num_seqs == 128
    assert candidate.judge.language_model_only is True
    assert candidate.judge.skip_mm_profiling is True


def test_text_only_controls_are_forwarded_to_vllm() -> None:
    kwargs = vllm_kwargs_from_config(
        model_name="Qwen/Qwen3.5-35B-A3B",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        max_model_len=4096,
        max_num_seqs=128,
        language_model_only=True,
        skip_mm_profiling=True,
        enforce_eager=False,
        enable_flashinfer_autotune=None,
        spec_method=None,
        spec_model=None,
        spec_tokens=None,
    )

    assert kwargs["language_model_only"] is True
    assert kwargs["skip_mm_profiling"] is True
    assert kwargs["enforce_eager"] is False
    assert "enable_flashinfer_autotune" not in kwargs


def test_eager_fallback_changes_only_runtime_mode() -> None:
    compiled = compose_ensemble(
        "blackbox_reasoning_ensemble_dks3072_logit_qwen35a3b_v1"
    )
    eager = compose_ensemble(
        "blackbox_reasoning_ensemble_dks3072_logit_qwen35a3b_eager_v1"
    )

    assert eager.judge.model == compiled.judge.model
    assert eager.judge.language_model_only == compiled.judge.language_model_only
    assert eager.judge.skip_mm_profiling == compiled.judge.skip_mm_profiling
    assert eager.judge.enforce_eager is True
    assert compiled.judge.enforce_eager is False
    assert eager.judge.enable_flashinfer_autotune is False
    assert OmegaConf.select(
        compiled, "judge.enable_flashinfer_autotune", default=None
    ) is None
    assert eager.ensemble.members == compiled.ensemble.members
    assert eager.ensemble.aggregation == compiled.ensemble.aggregation
    assert eager.scoring == compiled.scoring


def test_qwen35a3b_reasoning_gate_is_a_matched_27b_model_swap() -> None:
    dense = compose_ensemble(
        "blackbox_reasoning_qwen27b_ensemble_dks_member4096_v1"
    )
    sparse = compose_ensemble(
        "blackbox_reasoning_qwen35a3b_ensemble_dks_member4096_v1"
    )

    assert dense.judge.model == "Qwen/Qwen3.5-27B"
    assert sparse.judge.model == "Qwen/Qwen3.5-35B-A3B"
    assert sparse.judge.mode == dense.judge.mode == "generate"
    assert sparse.judge.max_tokens == dense.judge.max_tokens == 4096
    assert sparse.judge.max_prompt_chars == dense.judge.max_prompt_chars
    assert sparse.judge.temperature == dense.judge.temperature
    assert sparse.ensemble.members == dense.ensemble.members
    assert sparse.ensemble.order == dense.ensemble.order == "member"
    assert sparse.ensemble.aggregation == dense.ensemble.aggregation == "max"
    assert sparse.scoring == dense.scoring
    assert sparse.judge.language_model_only is True
    assert sparse.judge.skip_mm_profiling is True
    assert sparse.judge.enforce_eager is True
    assert sparse.judge.enable_flashinfer_autotune is False


def test_postreason_gate_reuses_frozen_reasoning_without_selected_label() -> None:
    generated = compose_ensemble(
        "blackbox_reasoning_qwen35a3b_ensemble_dks_member4096_v1"
    )
    rescored = compose_ensemble(
        "blackbox_reasoning_qwen35a3b_ensemble_dks_member4096_postreason_logits_v1"
    )

    assert rescored.judge.mode == "structured"
    assert rescored.judge.model == generated.judge.model
    assert rescored.ensemble == generated.ensemble
    assert rescored.scoring == generated.scoring
    assert rescored.judge.reasoning_cache_path.endswith(
        "qwen35a3b_reason_ensemble_dks_member4096_v1/validation/generations.jsonl"
    )


def test_qwen27_postreason_gate_is_the_matched_dense_teacher_rescore() -> None:
    generated = compose_ensemble(
        "blackbox_reasoning_qwen27b_ensemble_dks_member4096_v1"
    )
    rescored = compose_ensemble(
        "blackbox_reasoning_qwen27b_ensemble_dks_member4096_postreason_logits_v1"
    )

    assert rescored.judge.mode == "structured"
    assert rescored.judge.model == "Qwen/Qwen3.5-27B"
    assert rescored.judge.model == generated.judge.model
    assert rescored.ensemble == generated.ensemble
    assert rescored.scoring == generated.scoring
    assert rescored.judge.reasoning_cache_path.endswith(
        "qwen27b_reason_ensemble_dks_member4096_v1/validation/generations.jsonl"
    )


def test_terminal_generated_rating_is_removed_before_rescore() -> None:
    reasoning = "Check the claims carefully.\n\n**Final Rating:** **6**"
    stripped, rating = strip_terminal_generated_rating(
        reasoning,
        rating_min=1,
        rating_max=7,
    )

    assert stripped == "Check the claims carefully."
    assert rating == 6

    incomplete = "Still checking the evidence.\n\nRating"
    assert strip_terminal_generated_rating(
        incomplete,
        rating_min=1,
        rating_max=7,
    ) == (incomplete, None)


def test_soft_rating_distribution_is_normalized_and_auditable() -> None:
    assert normalize_rating_probs({1: 1.0, 2: 3.0}) == {1: 0.25, 2: 0.75}
    assert missing_requested_token_ids(
        {11: -0.1, 13: -2.0},
        [11, 12, 13],
    ) == [12]
