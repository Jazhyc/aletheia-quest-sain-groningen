#!/usr/bin/env python3
"""Validate and publish the selected Phoenix 8.1 BF16 LoRA package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kimi_liars_enrichment.export_bf16_adapter import (
    CANONICAL_PREFIX,
    sha256_file,
)


VISION_EXCLUDE_PATTERN = r".*(visual|vision_tower|merger|patch_embed).*"


def validate_adapter(
    adapter_dir: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Validate the exact deployment package before any external write."""
    required = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "bf16_export.json",
    }
    missing = sorted(name for name in required if not (adapter_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"incomplete adapter directory; missing {missing}")
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    expected = {
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "r": 16,
        "lora_alpha": 32,
        "exclude_modules": VISION_EXCLUDE_PATTERN,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unexpected adapter configuration: {mismatches}")

    weights_path = adapter_dir / "adapter_model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        dtypes = sorted({str(handle.get_tensor(key).dtype) for key in keys})
    if len(keys) != 256:
        raise ValueError(f"expected 256 LoRA tensors, found {len(keys)}")
    noncanonical = [key for key in keys if not key.startswith(CANONICAL_PREFIX)]
    if noncanonical:
        raise ValueError(f"noncanonical LoRA keys: {noncanonical[:3]}")
    if dtypes != ["torch.bfloat16"]:
        raise ValueError(f"expected BF16 LoRA tensors, found {dtypes}")
    weight_sha256 = sha256_file(weights_path)
    if expected_sha256 is not None and weight_sha256 != expected_sha256:
        raise ValueError(
            f"adapter digest mismatch: expected={expected_sha256} "
            f"actual={weight_sha256}"
        )
    return {
        "tensor_count": len(keys),
        "dtypes": dtypes,
        "weight_sha256": weight_sha256,
        "weight_bytes": weights_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--commit-message",
        default="Publish Phoenix 8.1 Kimi Liars adapter",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing from .env")
    validation = validate_adapter(
        args.adapter_dir.resolve(), args.expected_sha256
    )
    api = HfApi(token=token)
    account = str(api.whoami()["name"])
    if account.lower() != "jazhyc":
        raise RuntimeError(f"expected Jazhyc Hugging Face account, got {account!r}")
    api.create_repo(
        repo_id=args.repository,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=args.repository,
        repo_type="model",
        folder_path=args.adapter_dir,
        commit_message=args.commit_message,
    )
    info = api.model_info(
        args.repository,
        revision=commit.oid,
        files_metadata=True,
    )
    remote_weights = next(
        item
        for item in info.siblings
        if item.rfilename == "adapter_model.safetensors"
    )
    remote_lfs = getattr(remote_weights, "lfs", None) or {}
    if remote_lfs.get("sha256") != validation["weight_sha256"]:
        raise RuntimeError(
            f"remote digest mismatch: local={validation['weight_sha256']} "
            f"remote={remote_lfs.get('sha256')}"
        )
    print(
        f"repository={args.repository} revision={commit.oid} "
        f"tensors={validation['tensor_count']} dtypes={validation['dtypes']} "
        f"weight_sha256={validation['weight_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
