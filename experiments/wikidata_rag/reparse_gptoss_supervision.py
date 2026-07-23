#!/usr/bin/env python3
"""Reparse saved GPT-OSS completions after validator improvements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.wikidata_rag.label_gptoss_retrieval_candidates import (
    parse_json_array,
    structural_errors,
    validate_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    output = []
    for row in rows:
        labels, errors = validate_labels(parse_json_array(row.get("raw_completion", "")), row)
        structural = structural_errors(errors)
        output.append({
            **row, "labels": labels, "parse_error": bool(structural),
            "grounding_error": bool(errors) and not bool(structural),
            "validation_errors": errors,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n"
    )
    print(
        f"wrote {len(output)} rows; valid={sum(not row['parse_error'] for row in output)}; "
        f"grounding_warnings={sum(row['grounding_error'] for row in output)}"
    )


if __name__ == "__main__":
    main()
