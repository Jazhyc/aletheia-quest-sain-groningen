from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.blackbox.run_judge import vllm_kwargs_from_config


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
        spec_method=None,
        spec_model=None,
        spec_tokens=None,
    )

    assert kwargs["language_model_only"] is True
    assert kwargs["skip_mm_profiling"] is True
