from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fact_extractor_self_consistency.run_ndif_whole_output_self_query import (
    batch_size_for,
    encode_batches,
    parse_boolean,
    safe_group_name,
)


def test_family_batch_sizes() -> None:
    assert batch_size_for("Qwen/Qwen3.5-27B") == 16
    assert batch_size_for("google/gemma-3-27b-it") == 8
    assert batch_size_for("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16") == 1


def test_strict_boolean_parser() -> None:
    assert parse_boolean("TRUE") is True
    assert parse_boolean("false.\n") is False
    assert parse_boolean("The answer is TRUE") is None


def test_group_name_is_path_safe() -> None:
    name = safe_group_name("Qwen/Qwen3.5-27B", "aletheias-quest/a-mo")
    assert "/" not in name
    assert name == "Qwen_Qwen3.5-27B__aletheias-quest_a-mo"


class _Tokenizer:
    def __call__(self, prompts, **kwargs):
        return kwargs


def test_dynamic_and_fixed_padding_are_available() -> None:
    batch = encode_batches(_Tokenizer(), ["a", "b"], 8, padding_length=None)[0]
    assert batch["padding"] is True
    fixed = encode_batches(_Tokenizer(), ["a"], 16, padding_length=143)[0]
    assert fixed["padding"] == "max_length"
    assert fixed["max_length"] == 143
