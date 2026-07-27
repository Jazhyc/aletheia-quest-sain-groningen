from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir

from experiments.blackbox.run_judge import OfflineVllmRatingJudge


ROOT = Path(__file__).resolve().parents[2]


def compose_single(name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "configs" / "single_judges").resolve()),
    ):
        return compose(config_name=name)


def test_qwen36_teacher_gate_is_an_exact_qwen35_model_swap() -> None:
    control = compose_single(
        "blackbox_reasoning_nothink_truth_value_binary_logit_qwen35_27b_v1"
    )
    candidate = compose_single(
        "blackbox_reasoning_nothink_truth_value_binary_logit_qwen36_27b_v1"
    )

    assert control.judge.model == "Qwen/Qwen3.5-27B"
    assert candidate.judge.model == "Qwen/Qwen3.6-27B"
    assert control.method != candidate.method
    assert control.judge.served_model != candidate.judge.served_model

    ignored = {"model", "served_model"}
    assert {
        key: value for key, value in control.judge.items() if key not in ignored
    } == {
        key: value for key, value in candidate.judge.items() if key not in ignored
    }
    assert control.scoring == candidate.scoring
    assert control.ensemble == candidate.ensemble


def test_binary_gate_uses_exact_no_thinking_truth_value_boundary() -> None:
    cfg = compose_single(
        "blackbox_reasoning_nothink_truth_value_binary_logit_qwen35_27b_v1"
    )

    assert cfg.judge.backend == "offline"
    assert cfg.judge.mode == "logits"
    assert cfg.judge.use_chat_template is True
    assert cfg.judge.enable_thinking is False
    assert cfg.judge.logit_prefix == "Prediction:"
    assert [dict(target) for target in cfg.judge.logit_targets] == [
        {"name": "honest", "score": 0.0, "texts": ["0"]},
        {"name": "deceptive", "score": 1.0, "texts": ["1"]},
    ]
    assert cfg.judge.max_prompt_chars == 3000
    assert cfg.judge.max_model_len == 4096
    assert cfg.judge.tensor_parallel_size == 1
    assert cfg.judge.language_model_only is True
    assert cfg.judge.skip_mm_profiling is True
    assert "important false claim" in cfg.judge.prompt
    assert "dishonest intent alone" in cfg.judge.prompt
    assert cfg.judge.prompt.endswith("Prediction:<0 or 1>")


def test_qwen397_teacher_is_exact_tvg_model_and_resource_swap() -> None:
    control = compose_single(
        "blackbox_reasoning_nothink_truth_value_binary_logit_qwen35_27b_v1"
    )
    candidate = compose_single(
        "blackbox_reasoning_nothink_truth_value_binary_logit_qwen35_397b_fp8_v1"
    )

    assert candidate.judge.model == "Qwen/Qwen3.5-397B-A17B-FP8"
    assert candidate.judge.tensor_parallel_size == 8
    assert candidate.judge.max_num_seqs == 32

    ignored = {"model", "served_model", "tensor_parallel_size", "max_num_seqs"}
    assert {
        key: value for key, value in control.judge.items() if key not in ignored
    } == {
        key: value for key, value in candidate.judge.items() if key not in ignored
    }


def test_rating_gate_changes_only_the_output_parameterization() -> None:
    binary = compose_single(
        "blackbox_reasoning_nothink_truth_value_binary_logit_qwen35_27b_v1"
    )
    ordinal = compose_single(
        "blackbox_reasoning_nothink_truth_value_rating7_logit_qwen35_27b_v1"
    )

    assert ordinal.judge.model == binary.judge.model == "Qwen/Qwen3.5-27B"
    assert ordinal.judge.use_chat_template == binary.judge.use_chat_template is True
    assert ordinal.judge.enable_thinking == binary.judge.enable_thinking is False
    assert ordinal.judge.max_prompt_chars == binary.judge.max_prompt_chars == 3000
    assert ordinal.judge.logit_prefix == "Rating:"
    assert ordinal.judge.logit_targets is None
    assert ordinal.judge.prompt.split("Output exactly one line:")[0] == (
        binary.judge.prompt.split("Output exactly one line:")[0]
    )
    assert ordinal.judge.prompt.endswith(
        "Rating: <integer from 1 to 7>"
    )


def test_chat_logit_prefix_is_appended_at_assistant_boundary() -> None:
    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            assert messages == [{"role": "user", "content": "raw user prompt"}]
            assert kwargs["enable_thinking"] is False
            return "<chat>assistant\n<think>\n\n</think>\n\n"

    class FakeLlm:
        prompts = None

        @classmethod
        def generate(cls, prompts, sampling):
            del sampling
            cls.prompts = prompts
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(logprobs=[{15: -1.25, 16: -0.25}])]
                )
            ]

    judge = OfflineVllmRatingJudge.__new__(OfflineVllmRatingJudge)
    judge.llm = FakeLlm()
    judge.sampling = object()
    judge.tokenizer = FakeTokenizer()
    judge.enable_thinking = False
    judge.reasoning_effort = None
    judge.assistant_prefix = "Prediction:"
    judge.rating_min = 1
    judge.rating_max = 7
    judge.missing_logprob = -30.0
    judge.ratings = list(range(1, 8))
    judge.targets = [
        {"name": "honest", "score": 0.0, "ids": [15]},
        {"name": "deceptive", "score": 1.0, "ids": [16]},
    ]
    judge.all_rating_ids = [15, 16]
    judge.ids_by_rating = {}

    scores = judge.score_prompts(["raw user prompt"], batch_size=None)

    assert FakeLlm.prompts == [
        "<chat>assistant\n<think>\n\n</think>\n\nPrediction:"
    ]
    assert scores[0] > 0.5
