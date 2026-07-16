#!/usr/bin/env python3
"""Merge compatible LoRA adapters and truncate their summed delta to one rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors.torch import load_file, save_file


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def parse_weighted_adapter(value: str) -> tuple[float, Path]:
    """Parse WEIGHT:PATH without confusing absolute paths with the weight."""
    weight, separator, path = value.partition(":")
    if not separator or not path:
        raise argparse.ArgumentTypeError("adapter must be WEIGHT:PATH")
    try:
        parsed_weight = float(weight)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid adapter weight: {weight!r}") from error
    return parsed_weight, Path(path)


def truncated_product_factors(
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate ``left @ right`` using rank-limited factors without forming it."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError(
            f"incompatible low-rank factors: left={left.shape}, right={right.shape}"
        )
    if rank <= 0 or rank > min(left.shape[0], right.shape[1]):
        raise ValueError(f"invalid output rank {rank} for product {left.shape[0]}x{right.shape[1]}")

    # QR reduces the only SVD to the small summed-rank core. This is exact before
    # truncation even for the very wide Qwen projection matrices.
    q_left, r_left = torch.linalg.qr(left.float(), mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T.float(), mode="reduced")
    core = r_left @ r_right.T
    u, singular_values, vh = torch.linalg.svd(core, full_matrices=False)
    kept = min(rank, singular_values.numel())
    root = singular_values[:kept].sqrt()
    merged_left = (q_left @ u[:, :kept]) * root.unsqueeze(0)
    merged_right = root.unsqueeze(1) * (vh[:kept] @ q_right.T)
    if kept < rank:
        merged_left = torch.nn.functional.pad(merged_left, (0, rank - kept))
        merged_right = torch.nn.functional.pad(merged_right, (0, 0, 0, rank - kept))
    return merged_left, merged_right


def validate_configs(configs: list[dict[str, Any]]) -> int:
    """Return the shared rank after checking delta-scaling compatibility."""
    reference = configs[0]
    fields = (
        "base_model_name_or_path",
        "r",
        "lora_alpha",
        "use_rslora",
        "fan_in_fan_out",
    )
    for config in configs[1:]:
        mismatched = [field for field in fields if config.get(field) != reference.get(field)]
        if set(config.get("target_modules") or []) != set(reference.get("target_modules") or []):
            mismatched.append("target_modules")
        if mismatched:
            raise ValueError(f"incompatible adapter configs: {mismatched}")
    return int(reference["r"])


def merge_adapter_states(
    states: list[dict[str, torch.Tensor]],
    weights: list[float],
    rank: int,
) -> dict[str, torch.Tensor]:
    """Return a rank-limited weighted sum of compatible LoRA delta matrices."""
    if len(states) != len(weights) or not states:
        raise ValueError("states and weights must have the same non-zero length")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("adapter tensor keys differ")
    if any(not key.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX)) for key in keys):
        raise ValueError("only plain LoRA A/B adapter tensors are supported")

    merged: dict[str, torch.Tensor] = {}
    for a_key in sorted(key for key in keys if key.endswith(LORA_A_SUFFIX)):
        b_key = a_key.removesuffix(LORA_A_SUFFIX) + LORA_B_SUFFIX
        if b_key not in keys:
            raise ValueError(f"missing paired tensor for {a_key}")
        a_tensors = [state[a_key] for state in states]
        b_tensors = [state[b_key] for state in states]
        if any(tensor.shape != a_tensors[0].shape for tensor in a_tensors[1:]):
            raise ValueError(f"LoRA A shapes differ for {a_key}")
        if any(tensor.shape != b_tensors[0].shape for tensor in b_tensors[1:]):
            raise ValueError(f"LoRA B shapes differ for {b_key}")
        if a_tensors[0].shape[0] != rank or b_tensors[0].shape[1] != rank:
            raise ValueError(f"configured rank does not match tensors for {a_key}")

        # Each source has the same alpha/r scaling, so applying the mixture
        # weights to B gives the desired weighted sum before rank truncation.
        right = torch.cat([tensor.float() for tensor in a_tensors], dim=0)
        left = torch.cat(
            [weight * tensor.float() for weight, tensor in zip(weights, b_tensors, strict=True)],
            dim=1,
        )
        merged_b, merged_a = truncated_product_factors(left, right, rank)
        merged[a_key] = merged_a.to(a_tensors[0].dtype).contiguous()
        merged[b_key] = merged_b.to(b_tensors[0].dtype).contiguous()
    return merged


def copy_adapter_metadata(template: Path, output: Path) -> None:
    """Copy tokenizer and PEFT metadata while leaving weights to the merger."""
    output.mkdir(parents=True, exist_ok=True)
    for path in template.iterdir():
        if path.name == "adapter_model.safetensors" or not path.is_file():
            continue
        shutil.copy2(path, output / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        type=parse_weighted_adapter,
        help="weighted source as WEIGHT:PATH; repeat for each source",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--template-adapter",
        type=Path,
        help="metadata/prompt source; defaults to the final --adapter path",
    )
    args = parser.parse_args()

    weights = [weight for weight, _ in args.adapter]
    paths = [path.resolve() for _, path in args.adapter]
    configs = [json.loads((path / "adapter_config.json").read_text()) for path in paths]
    rank = validate_configs(configs)
    states = [load_file(path / "adapter_model.safetensors", device="cpu") for path in paths]
    merged = merge_adapter_states(states, weights, rank)

    output = args.output.resolve()
    template = (args.template_adapter or paths[-1]).resolve()
    copy_adapter_metadata(template, output)
    save_file(merged, output / "adapter_model.safetensors")
    source_config = template.parent / "config.yaml"
    if source_config.is_file():
        shutil.copy2(source_config, output.parent / "config.yaml")
    manifest = {
        "rank": rank,
        "sources": [
            {"weight": weight, "adapter": path.as_posix()}
            for weight, path in zip(weights, paths, strict=True)
        ],
        "template_adapter": template.as_posix(),
    }
    (output.parent / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved merged rank-{rank} adapter to {output}")


if __name__ == "__main__":
    main()
