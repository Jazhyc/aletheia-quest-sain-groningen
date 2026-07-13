#!/usr/bin/env python3
"""Pilot Wikidata fact retrieval against varied-deception training examples."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CACHE_VERSION = 2
SEARCH_URL = "https://en.wikipedia.org/w/api.php"
ENTITY_URL = "https://www.wikidata.org/w/api.php"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")

# Compact, generally useful properties. Keeping this explicit approximates the
# eventual submission-sized index better than retaining every Wikidata claim.
PROPERTY_LABELS = {
    "P17": "country", "P19": "place of birth", "P20": "place of death",
    "P27": "country of citizenship", "P30": "continent", "P31": "instance of",
    "P36": "capital", "P37": "official language", "P39": "position held",
    "P40": "child", "P50": "author", "P54": "member of sports team",
    "P57": "director", "P61": "discoverer or inventor", "P69": "educated at",
    "P101": "field of work", "P102": "political party", "P106": "occupation",
    "P108": "employer", "P112": "founded by", "P119": "place of burial",
    "P131": "located in", "P136": "genre", "P138": "named after",
    "P155": "follows", "P156": "followed by", "P159": "headquarters",
    "P161": "cast member", "P166": "award received", "P170": "creator",
    "P175": "performer", "P176": "manufacturer", "P178": "developer",
    "P179": "part of series", "P264": "record label", "P272": "producer",
    "P276": "location", "P279": "subclass of", "P286": "head coach",
    "P361": "part of", "P364": "original language", "P400": "platform",
    "P449": "original broadcaster", "P463": "member of", "P495": "country of origin",
    "P527": "has part", "P551": "residence", "P569": "date of birth",
    "P570": "date of death", "P571": "inception", "P576": "dissolved or abolished",
    "P577": "publication date", "P580": "start time", "P582": "end time",
    "P585": "point in time", "P740": "location of formation", "P800": "notable work",
    "P840": "narrative location", "P915": "filming location", "P937": "work location",
    "P1082": "population", "P1120": "number of deaths", "P2044": "elevation",
    "P2046": "area", "P2047": "duration", "P2120": "radius",
    "P2130": "cost", "P2142": "box office",
}

STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "and", "are", "assistant",
    "because", "been", "before", "being", "between", "both", "but", "can", "claim",
    "claims", "contains", "context", "correct", "deceptive", "does", "during", "each",
    "established", "fact", "facts", "false", "for", "from", "had", "has", "have",
    "important", "into", "its", "misleading", "more", "most", "not", "output", "response",
    "should", "states", "such", "than", "that", "the", "their", "there", "these", "they",
    "this", "through", "under", "user", "was", "were", "what", "when", "where", "which",
    "while", "with", "would",
}


def tokens(text: str) -> set[str]:
    return {
        token.lower().replace("’", "'")
        for token in TOKEN_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def extract_conversation(student_prompt: str) -> str:
    start = student_prompt.find("<context>")
    end = student_prompt.find("</context>", start + 9)
    if start < 0:
        return student_prompt
    return student_prompt[start + 9:end if end >= 0 else None].strip()


def search_queries(conversation: str, max_chars: int = 500) -> list[str]:
    """Build progressive MediaWiki queries without treating a whole chat as AND terms."""
    turns: dict[str, list[str]] = defaultdict(list)
    for match in re.finditer(
        r"(?:^|\n)(SYSTEM|USER|ASSISTANT):\s*(.*?)(?=\n(?:SYSTEM|USER|ASSISTANT):|\Z)",
        conversation,
        flags=re.DOTALL,
    ):
        turns[match.group(1)].append(" ".join(match.group(2).split()))
    candidates = []
    if turns["USER"]:
        candidates.append(turns["USER"][-1][-max_chars:])
    if turns["ASSISTANT"]:
        candidates.append(turns["ASSISTANT"][-1][:max_chars])
    final_exchange = " ".join(candidates)
    named_phrases = re.findall(
        r"(?:[A-Z][\w’'-]*)(?:\s+(?:of|the|de|van|von|[A-Z][\w’'-]*)){1,5}",
        final_exchange,
    )
    quoted = re.findall(r"[\"“‘']([^\"”’']{3,80})[\"”’']", final_exchange)
    candidates.extend(reversed(named_phrases[-4:]))
    candidates.extend(reversed(quoted[-2:]))
    return list(dict.fromkeys(query for query in candidates if query))


def request_json(session: Any, url: str, params: dict[str, Any], delay: float) -> dict[str, Any]:
    for attempt in range(7):
        response = session.get(url, params=params, timeout=45)
        if response.status_code not in {429, 502, 503, 504}:
            response.raise_for_status()
            if delay:
                time.sleep(delay)
            return response.json()
        retry = response.headers.get("Retry-After")
        time.sleep(float(retry) if retry else min(60.0, 2.0**attempt))
    response.raise_for_status()
    raise AssertionError("unreachable")


def search_qids(session: Any, query: str, limit: int, delay: float) -> list[dict[str, str]]:
    """Use Wikipedia search only for entity linking; retain no Wikipedia prose."""
    data = request_json(session, SEARCH_URL, {
        "action": "query", "generator": "search", "gsrsearch": query,
        "gsrlimit": limit, "gsrnamespace": 0, "prop": "pageprops",
        "ppprop": "wikibase_item", "format": "json", "formatversion": 2,
        "maxlag": 5,
    }, delay)
    pages = sorted(data.get("query", {}).get("pages", []), key=lambda row: row.get("index", 9999))
    return [
        {"qid": page["pageprops"]["wikibase_item"], "title": str(page.get("title", ""))}
        for page in pages if page.get("pageprops", {}).get("wikibase_item")
    ]


def fetch_entities(session: Any, qids: Iterable[str], delay: float) -> dict[str, dict[str, Any]]:
    qids = list(dict.fromkeys(qids))
    if not qids:
        return {}
    data = request_json(session, ENTITY_URL, {
        "action": "wbgetentities", "ids": "|".join(qids), "languages": "en",
        "languagefallback": 1, "props": "labels|aliases|descriptions|claims",
        "format": "json", "formatversion": 2, "maxlag": 5,
    }, delay)
    return data.get("entities", {})


def datavalue(snak: dict[str, Any]) -> tuple[str, str] | None:
    value = snak.get("datavalue", {}).get("value")
    kind = snak.get("datavalue", {}).get("type")
    if kind == "wikibase-entityid" and isinstance(value, dict) and value.get("id"):
        return "entity", value["id"]
    if kind == "time" and isinstance(value, dict):
        raw = str(value.get("time", "")).lstrip("+")
        precision = int(value.get("precision", 11))
        return "literal", raw[:4] if precision <= 9 else raw[:10]
    if kind == "quantity" and isinstance(value, dict):
        amount = str(value.get("amount", "")).lstrip("+")
        return "literal", amount
    if kind in {"string", "monolingualtext"}:
        return "literal", str(value.get("text", "") if isinstance(value, dict) else value)
    return None


def entity_stub(entity: dict[str, Any]) -> dict[str, Any]:
    label = entity.get("labels", {}).get("en", {}).get("value", "")
    aliases = [row.get("value", "") for row in entity.get("aliases", {}).get("en", [])]
    description = entity.get("descriptions", {}).get("en", {}).get("value", "")
    claims: list[dict[str, str]] = []
    for pid, statements in entity.get("claims", {}).items():
        if pid not in PROPERTY_LABELS:
            continue
        for statement in statements[:3]:
            if statement.get("rank") == "deprecated":
                continue
            parsed = datavalue(statement.get("mainsnak", {}))
            if parsed:
                kind, value = parsed
                claims.append({"property": PROPERTY_LABELS[pid], "kind": kind, "value": value})
    return {"label": label, "aliases": aliases[:8], "description": description, "claims": claims}


def resolve_records(session: Any, linked: list[dict[str, str]], delay: float) -> list[dict[str, Any]]:
    entities = fetch_entities(session, [row["qid"] for row in linked], delay)
    stubs = {qid: entity_stub(entity) for qid, entity in entities.items()}
    object_ids = {
        claim["value"] for stub in stubs.values() for claim in stub["claims"]
        if claim["kind"] == "entity"
    }
    objects = fetch_entities(session, sorted(object_ids), delay)
    object_labels = {
        qid: entity.get("labels", {}).get("en", {}).get("value", qid)
        for qid, entity in objects.items()
    }
    records = []
    for link in linked:
        stub = stubs.get(link["qid"])
        if not stub:
            continue
        facts = [
            f"{claim['property']}: {object_labels.get(claim['value'], claim['value'])}"
            if claim["kind"] == "entity" else f"{claim['property']}: {claim['value']}"
            for claim in stub.pop("claims")
        ]
        records.append({**link, **stub, "facts": facts[:24]})
    return records


def evidence_text(entities: list[dict[str, Any]]) -> str:
    return "\n".join(
        " | ".join(filter(None, [
            entity.get("label", ""), "; ".join(entity.get("aliases", [])),
            entity.get("description", ""), "; ".join(entity.get("facts", [])),
        ])) for entity in entities
    )


def score_record(record: dict[str, Any], evidence: str) -> dict[str, float]:
    conversation_tokens = tokens(record["conversation"])
    target_tokens = tokens(record["reasoning_summary"])
    evidence_tokens = tokens(evidence)
    novel_target = target_tokens - conversation_tokens
    return {
        "summary_recall": len(target_tokens & evidence_tokens) / max(1, len(target_tokens)),
        "novel_target_recall": len(novel_target & evidence_tokens) / max(1, len(novel_target)),
        "evidence_precision": len(target_tokens & evidence_tokens) / max(1, len(evidence_tokens)),
    }


def balanced_sample(records: list[dict[str, Any]], per_label: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "varied-deception" in record.get("dataset", "") and not record.get("parse_error", True):
            by_label[int(record["label"])].append(record)
    selected = []
    for label in (0, 1):
        selected.extend(rng.sample(by_label[label], min(per_label, len(by_label[label]))))
    rng.shuffle(selected)
    return selected


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else math.nan


def write_report(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["# Wikidata varied-training retrieval pilot", ""]
    scored = [row for row in records if not row.get("error")]
    lines += [f"Rows: {len(records)}; successful: {len(scored)}", ""]
    lines += ["| group | coverage | summary recall | novel target recall | shuffled novel recall | evidence precision |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, name in [(None, "all"), (0, "honest"), (1, "deceptive")]:
        subset = [row for row in scored if label is None or row["label"] == label]
        coverage = sum(bool(row["entities"]) for row in subset) / max(1, len(subset))
        lines.append(
            f"| {name} | {coverage:.3f} | {mean(subset, 'summary_recall'):.3f} | "
            f"{mean(subset, 'novel_target_recall'):.3f} | {mean(subset, 'shuffled_novel_target_recall'):.3f} | "
            f"{mean(subset, 'evidence_precision'):.3f} |"
        )
    lines += ["", "Novel target recall removes tokens already present in the conversation. The shuffled column scores another row's evidence as a generic-overlap baseline.", "", "## Audit examples", ""]
    ordered = sorted(scored, key=lambda row: row["novel_target_recall"], reverse=True)
    picks = (ordered[:3] + ordered[len(ordered)//2:len(ordered)//2+3] + ordered[-3:]) if ordered else []
    for row in picks:
        lines += [
            f"### label={row['label']} index={row['index']} novel_recall={row['novel_target_recall']:.3f}", "",
            f"Teacher summary: {row['reasoning_summary']}", "",
            "Retrieved:", "",
        ]
        for entity in row["entities"]:
            lines.append(f"- **{entity['label']}** ({entity['qid']}): {entity['description']}; " + "; ".join(entity["facts"][:8]))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-cache", type=Path, default=ROOT / "results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=50)
    parser.add_argument("--entities-per-row", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import requests

    source = [json.loads(line) for line in args.teacher_cache.read_text().splitlines() if line.strip()]
    selected = balanced_sample(source, args.per_label, args.seed)
    existing = {}
    if args.output.exists():
        existing = {(row["dataset"], row["index"]): row for row in map(json.loads, args.output.read_text().splitlines())}
    session = requests.Session()
    session.headers["User-Agent"] = "AletheiaWikidataPilot/0.1 (deception detection research)"
    if os.environ.get("WIKIMEDIA_ACCESS_TOKEN"):
        session.headers["Authorization"] = f"Bearer {os.environ['WIKIMEDIA_ACCESS_TOKEN']}"

    records = []
    for number, source_row in enumerate(selected, 1):
        key = (source_row["dataset"], source_row["index"])
        conversation = extract_conversation(source_row["student_prompt"])
        queries = search_queries(conversation)
        cached = existing.get(key)
        if (
            cached
            and cached.get("cache_version") == CACHE_VERSION
            and cached.get("queries") == queries
            and not cached.get("error")
        ):
            row = cached
        else:
            try:
                linked_by_qid = {}
                for query in queries:
                    for link in search_qids(session, query, args.entities_per_row, args.delay_seconds):
                        linked_by_qid.setdefault(link["qid"], link)
                    if len(linked_by_qid) >= args.entities_per_row:
                        break
                linked = list(linked_by_qid.values())[:args.entities_per_row]
                entities = resolve_records(session, linked, args.delay_seconds)
                error = None
            except Exception as exc:
                entities, error = [], f"{type(exc).__name__}: {exc}"
            row = {
                "dataset": source_row["dataset"], "index": source_row["index"],
                "cache_version": CACHE_VERSION,
                "label": int(source_row["label"]), "queries": queries,
                "conversation": conversation, "reasoning_summary": source_row["reasoning_summary"],
                "entities": entities, "error": error,
            }
        row.update(score_record(row, evidence_text(row["entities"])))
        records.append(row)
        if number % 10 == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("\n".join(json.dumps(item) for item in records) + "\n")
            print(f"processed {number}/{len(selected)}", flush=True)

    # Shuffle within label so label-dependent summary language cannot inflate or
    # suppress the null baseline. A circular shift is deterministic and has no
    # fixed points when each label has at least two rows.
    alternatives: dict[tuple[int, str, Any], str] = {}
    for label in (0, 1):
        group = [row for row in records if row["label"] == label]
        evidences = [evidence_text(row["entities"]) for row in group]
        shifted = evidences[1:] + evidences[:1]
        for row, evidence in zip(group, shifted, strict=True):
            alternatives[(label, row["dataset"], row["index"])] = evidence
    for row in records:
        alternative = alternatives[(row["label"], row["dataset"], row["index"])]
        row["shuffled_novel_target_recall"] = score_record(row, alternative)["novel_target_recall"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item) for item in records) + "\n")
    write_report(args.report, records)
    print(f"wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
