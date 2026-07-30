#!/usr/bin/env python3
"""Export a canonical FP32 Qwen3.5 LoRA as a BF16 inference package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


CANONICAL_PREFIX = "base_model.model.model.language_model.layers."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_metadata(path: Path) -> dict[str, str] | None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.metadata()


def export_bf16(source: Path, destination: Path) -> dict[str, Any]:
    """Copy adapter metadata and convert only LoRA tensor payloads to BF16."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination adapters must differ")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source adapter is missing {missing}")

    state = load_file(source / "adapter_model.safetensors", device="cpu")
    if len(state) != 256:
        raise ValueError(f"expected 256 LoRA tensors, found {len(state)}")
    noncanonical = [
        key for key in state if not key.startswith(CANONICAL_PREFIX)
    ]
    if noncanonical:
        raise ValueError(f"noncanonical tensor paths: {noncanonical[:3]}")
    source_dtypes = sorted({str(tensor.dtype) for tensor in state.values()})
    if source_dtypes != ["torch.float32"]:
        raise ValueError(f"expected FP32 master tensors, found {source_dtypes}")

    destination.mkdir(parents=True, exist_ok=True)
    for source_file in source.iterdir():
        if source_file.is_file() and source_file.name != "adapter_model.safetensors":
            shutil.copy2(source_file, destination / source_file.name)
    converted = {key: tensor.to(torch.bfloat16) for key, tensor in state.items()}
    output_weights = destination / "adapter_model.safetensors"
    save_file(
        converted,
        output_weights,
        metadata=adapter_metadata(source / "adapter_model.safetensors"),
    )
    reloaded = load_file(output_weights, device="cpu")
    if set(reloaded) != set(state):
        raise RuntimeError("tensor key set changed during BF16 export")
    if {str(tensor.dtype) for tensor in reloaded.values()} != {"torch.bfloat16"}:
        raise RuntimeError("exported adapter contains non-BF16 tensors")
    for key, source_tensor in state.items():
        if not torch.equal(reloaded[key].float(), source_tensor.bfloat16().float()):
            raise RuntimeError(f"BF16 conversion mismatch for {key}")

    report = {
        "source": source.as_posix(),
        "destination": destination.as_posix(),
        "tensor_count": len(state),
        "source_dtypes": source_dtypes,
        "destination_dtypes": ["torch.bfloat16"],
        "source_weight_sha256": sha256_file(
            source / "adapter_model.safetensors"
        ),
        "destination_weight_sha256": sha256_file(output_weights),
        "source_weight_bytes": (
            source / "adapter_model.safetensors"
        ).stat().st_size,
        "destination_weight_bytes": output_weights.stat().st_size,
    }
    (destination / "bf16_export.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_bf16(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
