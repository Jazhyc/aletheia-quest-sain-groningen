from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from experiments.kimi_liars_enrichment.export_bf16_adapter import (
    CANONICAL_PREFIX,
    export_bf16,
)


def test_export_bf16_preserves_keys_and_rounds_payloads(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "adapter_config.json").write_text("{}\n")
    (source / "README.md").write_text("adapter\n")
    state = {
        f"{CANONICAL_PREFIX}{index}.x": torch.tensor(
            [index + 0.12345], dtype=torch.float32
        )
        for index in range(256)
    }
    save_file(state, source / "adapter_model.safetensors", metadata={"x": "y"})

    report = export_bf16(source, destination)

    converted = load_file(destination / "adapter_model.safetensors")
    assert set(converted) == set(state)
    assert {tensor.dtype for tensor in converted.values()} == {torch.bfloat16}
    assert (destination / "README.md").read_text() == "adapter\n"
    assert report["tensor_count"] == 256
    assert report["destination_weight_bytes"] < report["source_weight_bytes"]
