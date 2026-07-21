#!/usr/bin/env python3
"""Publish the two deployed Phoenix 3.0 PEFT adapters to Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
REPOSITORIES = {
    "deception": "Jazhyc/aletheias-phoenix-v3-deception-r1",
    "resolved_intent": "Jazhyc/aletheias-phoenix-v3-resolved-intent-r1",
}


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing from .env")
    api = HfApi(token=token)
    account = api.whoami()["name"]
    if account.lower() != "jazhyc":
        raise RuntimeError(f"expected Jazhyc Hugging Face account, got {account!r}")
    for member, repo_id in REPOSITORIES.items():
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=ROOT / "submission/phoenix_wright_v3_adapters" / member,
            commit_message=f"Upload Phoenix Wright 3.0 {member} rank-1 adapter",
        )
        print(f"{member}: {repo_id} {commit.oid}")


if __name__ == "__main__":
    main()
