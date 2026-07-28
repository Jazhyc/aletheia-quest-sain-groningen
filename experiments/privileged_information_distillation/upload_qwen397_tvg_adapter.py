#!/usr/bin/env python3
"""Publish the canonical Qwen397 soft-distillation LoRA for Phoenix 6.3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = (
    ROOT
    / "results/blackbox/qwen9b_qwen397_tvg_binary_softonly_varied_v1/adapter"
)
REPOSITORY = "Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16"
MODEL_CARD = (
    ROOT
    / "experiments/privileged_information_distillation/"
    "qwen397_tvg_adapter_model_card.md"
)
CANONICAL_PREFIX = "base_model.model.model.language_model.layers."
VISION_EXCLUDE_PATTERN = r".*(visual|vision_tower|merger|patch_embed).*"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete adapter directory: {adapter_dir}")
    config = json.loads(config_path.read_text())
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
    return {
        "tensor_count": len(keys),
        "dtypes": dtypes,
        "weight_sha256": sha256_file(weights_path),
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
    with tempfile.TemporaryDirectory(prefix="phoenix63-adapter-") as temp_dir:
        staging = Path(temp_dir) / "adapter"
        shutil.copytree(ADAPTER_DIR, staging)
        shutil.copy2(MODEL_CARD, staging / "README.md")
        commit = api.upload_folder(
            repo_id=REPOSITORY,
            repo_type="model",
            folder_path=staging,
            commit_message="Upload Phoenix 6.3 Qwen397 soft-distillation adapter",
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
