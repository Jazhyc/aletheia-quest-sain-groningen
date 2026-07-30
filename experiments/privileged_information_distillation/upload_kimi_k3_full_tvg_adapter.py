#!/usr/bin/env python3
"""Publish the canonical full-data Kimi K3 LoRA used by Phoenix 8."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "submission/phoenix_wright_adapters/main"
REPOSITORY = "Jazhyc/aletheias-phoenix-v8-kimi-k3-tvg-soft-full-r16-ep2"
CANONICAL_PREFIX = "base_model.model.model.language_model.layers."
VISION_EXCLUDE_PATTERN = r".*(visual|vision_tower|merger|patch_embed).*"
EXPECTED_WEIGHT_SHA256 = (
    "c3be0b58b5caf5750b3dea06b5a1490cb735483adaba51f6f09568054531edc0"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter(adapter_dir: Path) -> dict[str, Any]:
    required = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
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
    if dtypes != ["torch.float32"]:
        raise ValueError(f"expected float32 LoRA tensors, found {dtypes}")

    weight_sha256 = sha256_file(weights_path)
    if weight_sha256 != EXPECTED_WEIGHT_SHA256:
        raise ValueError(
            "unexpected adapter weight digest: "
            f"expected={EXPECTED_WEIGHT_SHA256} actual={weight_sha256}"
        )
    return {
        "tensor_count": len(keys),
        "dtypes": dtypes,
        "weight_sha256": weight_sha256,
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing from .env")

    validation = validate_adapter(ADAPTER_DIR)
    api = HfApi(token=token)
    account = str(api.whoami()["name"])
    if account.lower() != "jazhyc":
        raise RuntimeError(
            f"expected Jazhyc Hugging Face account, got {account!r}"
        )
    api.create_repo(
        repo_id=REPOSITORY,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=REPOSITORY,
        repo_type="model",
        folder_path=ADAPTER_DIR,
        commit_message="Publish Phoenix 8 full-data Kimi adapter",
    )

    info = api.model_info(
        REPOSITORY,
        revision=commit.oid,
        files_metadata=True,
    )
    weights = next(
        item
        for item in info.siblings
        if item.rfilename == "adapter_model.safetensors"
    )
    remote_lfs = getattr(weights, "lfs", None) or {}
    remote_sha256 = remote_lfs.get("sha256")
    if remote_sha256 != validation["weight_sha256"]:
        raise RuntimeError(
            "remote adapter digest mismatch: "
            f"local={validation['weight_sha256']} remote={remote_sha256}"
        )
    print(
        f"repository={REPOSITORY} revision={commit.oid} "
        f"tensors={validation['tensor_count']} dtypes={validation['dtypes']} "
        f"weight_sha256={validation['weight_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
