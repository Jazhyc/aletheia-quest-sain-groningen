#!/usr/bin/env python3
"""Measure NNsight payloads for the two standard HF string-stop paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nnsight.intervention import serialization
from transformers import AutoTokenizer
from transformers.generation.stopping_criteria import StopStringCriteria

from experiments.qwen27_ndif_deployment.contract import MODEL, STOP_STRINGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer_payload = serialization.dumps(tokenizer)
    restored = serialization.loads(tokenizer_payload)
    probe = "Rating: 7"
    if restored.encode(probe, add_special_tokens=False) != tokenizer.encode(
        probe, add_special_tokens=False
    ):
        raise RuntimeError("tokenizer serialization round trip changed encoding")

    criterion = StopStringCriteria(tokenizer, list(STOP_STRINGS))
    criterion_payload = serialization.dumps(criterion)
    result = {
        "model": args.model,
        "stop_strings": list(STOP_STRINGS),
        "tokenizer_payload_bytes": len(tokenizer_payload),
        "criterion_payload_bytes": len(criterion_payload),
        "criterion_embedding_shape": list(criterion.embedding_vec.shape),
        "tokenizer_roundtrip": True,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
