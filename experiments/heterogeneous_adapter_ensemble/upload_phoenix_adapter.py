#!/usr/bin/env python3
"""Publish the deployed Phoenix main adapter to Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "submission/phoenix_wright_adapters/main"
REPOSITORY = "Jazhyc/aletheias-phoenix-luna-medium-reasoning-r16"


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing from .env")
    api = HfApi(token=token)
    account = api.whoami()["name"]
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
        commit_message="Upload Luna medium reasoning adapter",
    )
    print(f"{REPOSITORY} {commit.oid}")


if __name__ == "__main__":
    main()
