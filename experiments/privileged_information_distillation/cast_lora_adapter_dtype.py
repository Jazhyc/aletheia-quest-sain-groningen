#!/usr/bin/env python3
"""Copy a PEFT adapter while casting every floating-point weight tensor."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def cast_adapter_weights(
    source: Path,
    output: Path,
    *,
    dtype: torch.dtype,
) -> dict[str, int]:
    """Copy an adapter directory and cast its floating-point safe tensors."""
    if source.resolve() == output.resolve():
        raise ValueError("source and output adapter directories must differ")
    if output.exists():
        raise FileExistsError(f"output adapter directory already exists: {output}")
    weights_path = source / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"adapter weights not found: {weights_path}")

    output.mkdir(parents=True)
    for path in source.iterdir():
        if path.name == weights_path.name:
            continue
        destination = output / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        elif path.is_file():
            shutil.copy2(path, destination)

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {}
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensors[key] = (
                tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
            )
    save_file(
        tensors,
        output / weights_path.name,
        metadata=metadata,
    )
    return {
        "tensors": len(tensors),
        "floating_tensors": sum(tensor.is_floating_point() for tensor in tensors.values()),
        "bytes": (output / weights_path.name).stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    args = parser.parse_args()

    result = cast_adapter_weights(
        args.source,
        args.output,
        dtype=DTYPES[args.dtype],
    )
    print(
        f"cast adapter to {args.dtype}: tensors={result['tensors']} "
        f"floating_tensors={result['floating_tensors']} bytes={result['bytes']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
