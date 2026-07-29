from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments.privileged_information_distillation.cast_lora_adapter_dtype import (
    cast_adapter_weights,
)


def test_cast_adapter_weights_copies_metadata_and_casts_floats(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "adapter_config.json").write_text('{"r": 32}\n')
    save_file(
        {
            "float": torch.tensor([[1.25, -2.5]], dtype=torch.float32),
            "integer": torch.tensor([3], dtype=torch.int64),
        },
        source / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )

    result = cast_adapter_weights(source, output, dtype=torch.bfloat16)
    tensors = load_file(output / "adapter_model.safetensors")

    assert tensors["float"].dtype == torch.bfloat16
    assert tensors["integer"].dtype == torch.int64
    assert (output / "adapter_config.json").read_text() == '{"r": 32}\n'
    assert result["tensors"] == 2
    assert result["floating_tensors"] == 1


def test_cast_adapter_weights_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    save_file(
        {"float": torch.tensor([1.0])},
        source / "adapter_model.safetensors",
    )

    with pytest.raises(FileExistsError):
        cast_adapter_weights(source, output, dtype=torch.bfloat16)
