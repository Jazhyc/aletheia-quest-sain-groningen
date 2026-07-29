import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from experiments.heterogeneous_adapter_ensemble.migrate_qwen35_peft_paths import (
    CANONICAL_PREFIX,
    LEGACY_PREFIX,
    VISION_EXCLUDE_PATTERN,
    canonical_adapter_key,
    convert_adapter_directory,
)


def write_adapter(path: Path, key: str) -> torch.Tensor:
    path.mkdir(parents=True)
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    (path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "exclude_modules": None,
    }))
    save_file(
        {key: tensor},
        path / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    return tensor


def test_canonical_adapter_key_inserts_language_model_once() -> None:
    legacy = LEGACY_PREFIX + "layers.0.self_attn.q_proj.lora_A.weight"
    canonical = CANONICAL_PREFIX + "layers.0.self_attn.q_proj.lora_A.weight"

    assert canonical_adapter_key(legacy) == (canonical, True)
    assert canonical_adapter_key(canonical) == (canonical, False)


def test_convert_adapter_directory_preserves_tensor_payload(tmp_path: Path) -> None:
    legacy = LEGACY_PREFIX + "layers.0.self_attn.q_proj.lora_A.weight"
    canonical = CANONICAL_PREFIX + "layers.0.self_attn.q_proj.lora_A.weight"
    source = tmp_path / "source"
    expected = write_adapter(source, legacy)

    result = convert_adapter_directory(source, tmp_path / "converted")

    converted = load_file(
        tmp_path / "converted/adapter_model.safetensors",
        device="cpu",
    )
    config = json.loads(
        (tmp_path / "converted/adapter_config.json").read_text()
    )
    assert list(converted) == [canonical]
    assert torch.equal(converted[canonical], expected)
    assert config["exclude_modules"] == VISION_EXCLUDE_PATTERN
    assert result["renamed_tensor_count"] == 1
    assert result["tensor_count"] == 1


def test_convert_adapter_directory_is_idempotent(tmp_path: Path) -> None:
    canonical = CANONICAL_PREFIX + "layers.0.mlp.up_proj.lora_B.weight"
    source = tmp_path / "source"
    write_adapter(source, canonical)

    result = convert_adapter_directory(source, tmp_path / "converted")

    assert result["renamed_tensor_count"] == 0
    assert result["weights_changed"] is False
