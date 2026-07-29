#!/usr/bin/env python3
"""Migrate text-only Qwen3.5 PEFT checkpoints to NDIF's canonical model tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    hf_hub_download,
)
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = (
    ROOT / "results/blackbox/qwen35_peft_path_migration_20260728"
)
LEGACY_PREFIX = "base_model.model.model."
CANONICAL_PREFIX = "base_model.model.model.language_model."
VISION_EXCLUDE_PATTERN = r".*(visual|vision_tower|merger|patch_embed).*"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_adapter_key(key: str) -> tuple[str, bool]:
    """Insert Qwen3.5's canonical ``language_model`` component when needed."""
    if key.startswith(CANONICAL_PREFIX):
        return key, False
    if key.startswith(LEGACY_PREFIX):
        return CANONICAL_PREFIX + key.removeprefix(LEGACY_PREFIX), True
    raise ValueError(f"unexpected adapter tensor key: {key!r}")


def canonicalize_config(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Exclude the vision tower when PEFT discovers decoder target modules."""
    updated = dict(config)
    changed = updated.get("exclude_modules") != VISION_EXCLUDE_PATTERN
    updated["exclude_modules"] = VISION_EXCLUDE_PATTERN
    return updated, changed


def adapter_metadata(path: Path) -> dict[str, str] | None:
    """Read safetensors metadata without materializing tensor payloads."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.metadata()


def convert_adapter_directory(source: Path, destination: Path) -> dict[str, Any]:
    """Convert and verify one adapter directory without changing the source."""
    source_config_path = source / "adapter_config.json"
    source_weights_path = source / "adapter_model.safetensors"
    if not source_config_path.is_file() or not source_weights_path.is_file():
        raise FileNotFoundError(
            f"{source} must contain adapter_config.json and "
            "adapter_model.safetensors"
        )

    config = json.loads(source_config_path.read_text())
    base_model = str(config.get("base_model_name_or_path", ""))
    if base_model not in {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"}:
        raise ValueError(f"unsupported base model {base_model!r} in {source}")

    state = load_file(source_weights_path, device="cpu")
    converted_state: dict[str, torch.Tensor] = {}
    renamed = 0
    for old_key, tensor in state.items():
        new_key, changed = canonical_adapter_key(old_key)
        if new_key in converted_state:
            raise ValueError(f"duplicate converted tensor key: {new_key}")
        converted_state[new_key] = tensor
        renamed += int(changed)

    key_count = len(state)
    if renamed not in {0, key_count}:
        raise ValueError(
            f"mixed legacy/canonical checkpoint: renamed {renamed}/{key_count}"
        )

    converted_config, config_changed = canonicalize_config(config)
    destination.mkdir(parents=True, exist_ok=True)
    converted_config_path = destination / "adapter_config.json"
    converted_weights_path = destination / "adapter_model.safetensors"
    converted_config_path.write_text(
        json.dumps(converted_config, indent=2, sort_keys=True) + "\n"
    )
    save_file(
        converted_state,
        converted_weights_path,
        metadata=adapter_metadata(source_weights_path),
    )

    reloaded = load_file(converted_weights_path, device="cpu")
    if set(reloaded) != set(converted_state):
        raise RuntimeError("converted checkpoint key set changed after saving")
    for new_key, converted_tensor in converted_state.items():
        if not torch.equal(reloaded[new_key], converted_tensor):
            raise RuntimeError(f"tensor payload changed for {new_key}")

    return {
        "base_model": base_model,
        "tensor_count": key_count,
        "renamed_tensor_count": renamed,
        "config_changed": config_changed,
        "weights_changed": bool(renamed),
        "old_weight_sha256": sha256_file(source_weights_path),
        "new_weight_sha256": sha256_file(converted_weights_path),
    }


def discover_repositories(api: HfApi, account: str) -> list[str]:
    """Find the account's Aletheia model repositories with PEFT configs."""
    repositories = []
    for model in api.list_models(author=account):
        repo_id = model.id
        if "aletheia" not in repo_id.lower():
            continue
        try:
            info = api.model_info(repo_id)
        except Exception as error:
            raise RuntimeError(f"could not inspect {repo_id}") from error
        if any(item.rfilename == "adapter_config.json" for item in info.siblings):
            repositories.append(repo_id)
    return sorted(repositories)


def download_adapter(
    repo_id: str,
    revision: str,
    destination: Path,
    token: str,
) -> None:
    """Download only the two files changed by the migration."""
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        downloaded = Path(hf_hub_download(
            repo_id,
            filename,
            revision=revision,
            token=token,
            local_dir=destination,
        ))
        target = destination / filename
        if downloaded.resolve() != target.resolve():
            shutil.copy2(downloaded, target)


def upload_conversion(
    api: HfApi,
    repo_id: str,
    parent_commit: str,
    converted_dir: Path,
) -> str:
    """Commit converted config and weights with an optimistic parent check."""
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        parent_commit=parent_commit,
        commit_message=(
            "Migrate Qwen3.5 LoRA paths for canonical NDIF model loading"
        ),
        operations=[
            CommitOperationAdd(
                path_in_repo="adapter_config.json",
                path_or_fileobj=converted_dir / "adapter_config.json",
            ),
            CommitOperationAdd(
                path_in_repo="adapter_model.safetensors",
                path_or_fileobj=converted_dir / "adapter_model.safetensors",
            ),
        ],
    )
    return commit.oid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--repo-id",
        action="append",
        help="migrate only this repository; repeat to select several",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="commit verified conversions; default is a local dry run",
    )
    parser.add_argument(
        "--local-dir",
        action="append",
        type=Path,
        help="also convert a local adapter directory in place",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="skip Hugging Face discovery and process only --local-dir values",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token and not args.local_only:
        raise SystemExit("HF_TOKEN is missing from .env")

    api = HfApi(token=token)
    account = None if args.local_only else str(api.whoami()["name"])
    repositories = []
    if not args.local_only:
        repositories = (
            sorted(set(args.repo_id))
            if args.repo_id
            else discover_repositories(api, str(account))
        )
    if not repositories and not args.local_dir:
        raise RuntimeError("no adapter repositories or local directories selected")

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "account": account,
        "legacy_prefix": LEGACY_PREFIX,
        "canonical_prefix": CANONICAL_PREFIX,
        "vision_exclude_pattern": VISION_EXCLUDE_PATTERN,
        "upload_requested": bool(args.upload),
        "repositories": [],
        "local_directories": [],
    }

    prepared: list[tuple[str, str, Path, dict[str, Any]]] = []
    for repo_id in repositories:
        info = api.model_info(repo_id, files_metadata=True)
        source_dir = work_dir / "repositories" / repo_id.replace("/", "--") / "source"
        converted_dir = source_dir.parent / "converted"
        download_adapter(repo_id, info.sha, source_dir, token)
        conversion = convert_adapter_directory(source_dir, converted_dir)
        record = {
            "repo_id": repo_id,
            "old_revision": info.sha,
            **conversion,
        }
        manifest["repositories"].append(record)
        prepared.append((repo_id, info.sha, converted_dir, record))
        print(
            f"prepared {repo_id}: renamed="
            f"{conversion['renamed_tensor_count']}/{conversion['tensor_count']} "
            f"old={conversion['old_weight_sha256']} "
            f"new={conversion['new_weight_sha256']}",
            flush=True,
        )

    if args.upload:
        for repo_id, old_revision, converted_dir, record in prepared:
            if not record["weights_changed"] and not record["config_changed"]:
                record["new_revision"] = old_revision
                record["upload_skipped"] = True
                continue
            new_revision = upload_conversion(
                api,
                repo_id,
                old_revision,
                converted_dir,
            )
            record["new_revision"] = new_revision
            record["upload_skipped"] = False
            print(f"uploaded {repo_id}: {new_revision}", flush=True)

    for local_dir_arg in args.local_dir or []:
        local_dir = local_dir_arg.resolve()
        converted_dir = work_dir / "local" / local_dir.name
        conversion = convert_adapter_directory(local_dir, converted_dir)
        backup_dir = work_dir / "local_backups" / local_dir.name
        backup_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            backup_path = backup_dir / filename
            if not backup_path.exists():
                shutil.copy2(local_dir / filename, backup_path)
            shutil.copy2(converted_dir / filename, local_dir / filename)
        manifest["local_directories"].append({
            "path": str(local_dir),
            "backup": str(backup_dir),
            **conversion,
        })
        print(f"converted local adapter {local_dir}", flush=True)

    manifest_name = "local_manifest.json" if args.local_only else "manifest.json"
    manifest_path = work_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
