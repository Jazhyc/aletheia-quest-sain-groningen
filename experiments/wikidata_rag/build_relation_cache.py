#!/usr/bin/env python3
"""Build a frozen cache for compact direct or bidirectional relation retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from experiments.wikidata_rag.claim_retrieval import extract_claim_query
from experiments.wikidata_rag.relation_retrieval import (
    retrieve_card_verdict,
    retrieve_relation_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity-database", type=Path, required=True)
    parser.add_argument("--relation-database", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument(
        "--require-verdict", action="store_true",
        help="Emit only support/counterevidence facts; abstain on uncertain relations.",
    )
    parser.add_argument(
        "--card-fallback", action="store_true",
        help="When structured retrieval abstains, use strict facts from the entity cards.",
    )
    args = parser.parse_args()
    entity = sqlite3.connect(f"file:{args.entity_database.resolve()}?mode=ro", uri=True)
    relation = sqlite3.connect(f"file:{args.relation_database.resolve()}?mode=ro", uri=True)
    source = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    output = []
    for number, record in enumerate(source, 1):
        result = retrieve_relation_evidence(
            entity, relation, record["conversation"], allow_inverse=not args.direct_only
        )
        used_card_fallback = False
        if args.card_fallback and result["status"] == "uncertain":
            fallback = retrieve_card_verdict(
                entity, extract_claim_query(record["conversation"]), result["subject_qids"]
            )
            if fallback:
                result["status"] = fallback["status"]
                result["facts"] = [row["fact"] for row in fallback["facts"]]
                result["fact_rows"] = fallback["facts"]
                result["abstain_reason"] = None
                used_card_fallback = True
        passages = []
        if result["facts"] and not (
            args.require_verdict and result["status"] == "uncertain"
        ):
            subjects = list(dict.fromkeys(
                str(row.get("subject_label", ""))
                for row in result["fact_rows"] if row.get("subject_label")
            ))
            passages = [{
                "title": " / ".join(subjects[:3]) or "Wikidata relation evidence",
                "text": "; ".join(result["facts"]),
            }]
        output.append({
            "dataset": record["dataset"], "index": record["index"],
            "conversation": record["conversation"], "claim": result["claim"],
            "real_passages": passages, "passages": passages,
            "entities": [], "qids": result["subject_qids"],
            "relation_status": result["status"],
            "answer_qids": result["answer_qids"],
            "used_inverse": result["used_inverse"],
            "used_two_hop": result.get("used_two_hop", False),
            "used_card_fallback": used_card_fallback,
            "relation_fact_rows": result["fact_rows"],
            "abstain_reason": (
                "uncertain_relation" if result["facts"] and not passages
                else result["abstain_reason"]
            ),
        })
        if number % 100 == 0:
            print(f"processed {number}/{len(source)}", flush=True)
    entity.close()
    relation.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in output) + "\n")
    covered = sum(bool(row["real_passages"]) for row in output)
    print(f"wrote {len(output)} rows; covered={covered}; abstained={len(output)-covered}")


if __name__ == "__main__":
    main()
