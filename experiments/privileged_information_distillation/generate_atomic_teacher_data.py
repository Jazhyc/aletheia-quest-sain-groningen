#!/usr/bin/env python3
"""Generate two-stage blind atomic-audit targets for privileged distillation."""

from __future__ import annotations

import html
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.core import (
    extract_harmony_final,
)
from experiments.privileged_information_distillation.generate_teacher_data import (
    load_teacher_rows,
    render_chat_prompt,
)


CLAIM_RE = re.compile(r"(?is)<claim>\s*(.*?)\s*</claim>")
QUOTE_RE = re.compile(r"(?is)<quote>\s*(.*?)\s*</quote>")
PROPOSITION_RE = re.compile(r"(?is)<proposition>\s*(.*?)\s*</proposition>")
ASSESSMENT_RE = re.compile(
    r"(?is)<assessment>\s*(true|false|uncertain)\s*</assessment>"
)
CONTEXT_CLAIM_RE = re.compile(r"(?is)<context_claim>\s*(.*?)\s*</context_claim>")
RELATION_RE = re.compile(
    r"(?is)<audit_label_relation>\s*(supports|conflicts|insufficient)\s*"
    r"</audit_label_relation>"
)
SUMMARY_RE = re.compile(
    r"(?is)<reasoning_summary>\s*(.*?)\s*</reasoning_summary>"
)
PREDICTION_RE = re.compile(r"(?im)\bPrediction\s*:\s*([01])\b")


def compact(text: str) -> str:
    """Collapse whitespace in a generated explanatory field."""
    return " ".join(str(text).split())


def output_from_student_prompt(student_prompt: str) -> str:
    """Recover the final assistant output from the canonical student prompt."""
    _, separator, remainder = student_prompt.partition("\n\n<output>\n")
    if not separator:
        raise ValueError("student prompt is missing <output>")
    output, separator, _ = remainder.rpartition("\n</output>")
    if not separator:
        raise ValueError("student prompt is missing </output>")
    return output


def replace_instruction(student_prompt: str, instruction: str) -> str:
    """Preserve rendered evidence fields while replacing judge instructions."""
    _, separator, evidence = student_prompt.partition("<context>")
    if not separator:
        raise ValueError("student prompt is missing <context>")
    return f"{instruction.strip()}\n\n<context>{evidence}"


def parse_blind_audit(raw_completion: str, output: str) -> dict[str, Any]:
    """Parse an audit and drop every claim without an exact source quote."""
    final = extract_harmony_final(raw_completion)
    claims: list[dict[str, str]] = []
    ungrounded_claims: list[dict[str, str]] = []
    malformed_claims = 0
    for body in CLAIM_RE.findall(final):
        quote_match = QUOTE_RE.search(body)
        proposition_match = PROPOSITION_RE.search(body)
        assessment_match = ASSESSMENT_RE.search(body)
        if not (quote_match and proposition_match and assessment_match):
            malformed_claims += 1
            continue
        claim = {
            "quote": html.unescape(quote_match.group(1).strip()),
            "proposition": compact(html.unescape(proposition_match.group(1))),
            "assessment": assessment_match.group(1).lower(),
        }
        if claim["quote"] and claim["proposition"] and claim["quote"] in output:
            claims.append(claim)
        else:
            ungrounded_claims.append(claim)
    context_claims = [
        compact(html.unescape(value))
        for value in CONTEXT_CLAIM_RE.findall(final)
        if compact(html.unescape(value))
    ]
    structure_valid = all(
        marker in final
        for marker in (
            "<factual_claims>",
            "</factual_claims>",
            "<context_claims>",
            "</context_claims>",
        )
    )
    return {
        "claims": claims,
        "context_claims": context_claims[:2],
        "ungrounded_claims": ungrounded_claims,
        "malformed_claims": malformed_claims,
        "structure_valid": structure_valid,
        "harmony_final": final,
    }


def xml_text(text: str) -> str:
    """Escape a validated field for deterministic XML-like serialization."""
    return html.escape(str(text), quote=False)


def format_frozen_audit(audit: dict[str, Any]) -> str:
    """Serialize only grounded claims; this immutable block is shared by both stages."""
    lines = ["<factual_claims>"]
    for claim in audit["claims"]:
        lines.extend((
            "<claim>",
            f"<quote>{xml_text(claim['quote'])}</quote>",
            f"<proposition>{xml_text(claim['proposition'])}</proposition>",
            f"<assessment>{claim['assessment']}</assessment>",
            "</claim>",
        ))
    lines.extend(("</factual_claims>", "<context_claims>"))
    for context_claim in audit["context_claims"]:
        lines.append(f"<context_claim>{xml_text(context_claim)}</context_claim>")
    lines.append("</context_claims>")
    return "\n".join(lines)


def build_reconcile_prompt(
    student_prompt: str,
    frozen_audit: str,
    label: int,
    template: str,
) -> str:
    """Expose the label only after the blind audit has been frozen."""
    instruction = template.replace("__GROUND_TRUTH__", str(int(label))).strip()
    _, separator, evidence = student_prompt.partition("<context>")
    if not separator:
        raise ValueError("student prompt is missing <context>")
    return (
        f"{instruction}\n\n<frozen_blind_audit>\n{frozen_audit}\n"
        f"</frozen_blind_audit>\n\n<context>{evidence}"
    )


def parse_reconciliation(
    raw_completion: str,
    expected_prediction: int,
) -> dict[str, Any] | None:
    """Parse label reconciliation without permitting claim rewrites."""
    final = extract_harmony_final(raw_completion)
    relation_matches = RELATION_RE.findall(final)
    summary_matches = SUMMARY_RE.findall(final)
    prediction_matches = PREDICTION_RE.findall(final)
    if not relation_matches or not summary_matches:
        return None
    summary = compact(summary_matches[-1])
    if not summary:
        return None
    prediction = (
        int(prediction_matches[-1])
        if prediction_matches
        else int(expected_prediction)
    )
    return {
        "audit_label_relation": relation_matches[-1].lower(),
        "reasoning_summary": summary,
        "prediction": prediction,
        "prediction_source": (
            "teacher_final" if prediction_matches else "privileged_label_fallback"
        ),
        "harmony_final": final,
    }


def format_atomic_student_target(
    frozen_audit: str,
    relation: str,
    summary: str,
    prediction: int,
) -> str:
    """Compose the SFT target from frozen blind evidence and label reconciliation."""
    return (
        f"{frozen_audit}\n"
        f"<audit_label_relation>{relation}</audit_label_relation>\n"
        f"<reasoning_summary>\n{summary.strip()}\n</reasoning_summary>\n"
        f"Prediction:{int(prediction)}"
    )


def load_jsonl(path: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[(record["dataset"], record["index"])] = record
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def generate_texts(
    llm: Any,
    prompts: list[str],
    sampling: Any,
    batch_size: int | None,
) -> list[str]:
    """Generate deterministic text in bounded batches."""
    outputs: list[str] = []
    width = len(prompts) if batch_size is None else int(batch_size)
    for start in range(0, len(prompts), max(1, width)):
        generated = llm.generate(prompts[start:start + width], sampling)
        batch = [item.outputs[0].text if item.outputs else "" for item in generated]
        outputs.extend(batch)
    return outputs


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="privileged_information_distillation_atomic_audit",
)
def main(cfg: DictConfig) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    root = Path(get_original_cwd()).resolve()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    rows = load_teacher_rows(cfg, root)
    print(f"loaded {len(rows)} two-stage teacher examples", flush=True)

    blind_path = Path(str(cfg.atomic_audit.blind_artifact))
    final_path = Path(str(cfg.teacher.artifact))
    if not blind_path.is_absolute():
        blind_path = root / blind_path
    if not final_path.is_absolute():
        final_path = root / final_path
    force = bool(cfg.teacher.force_regenerate)
    blind_cache = {} if force else load_jsonl(blind_path)
    final_cache = {} if force else load_jsonl(final_path)

    blind_rows: list[dict[str, Any]] = []
    missing_blind: list[dict[str, Any]] = []
    for row in rows:
        blind_prompt = replace_instruction(
            row["student_prompt"], str(cfg.atomic_audit.blind_prompt)
        )
        cached = blind_cache.get((row["dataset"], row["index"]))
        if (
            cached
            and cached.get("blind_prompt") == blind_prompt
            and cached.get("student_prompt") == row["student_prompt"]
            and cached.get("structure_valid") is True
            and cached.get("frozen_audit")
            and isinstance(cached.get("ungrounded_claims"), list)
        ):
            blind_rows.append(cached)
        else:
            missing_blind.append({**row, "blind_prompt": blind_prompt})
    print(
        f"blind cache hits={len(blind_rows)} generation required={len(missing_blind)}",
        flush=True,
    )

    tokenizer = None
    llm = None
    sampling = SamplingParams(
        max_tokens=int(cfg.teacher.max_tokens),
        temperature=float(cfg.teacher.temperature),
    )
    if missing_blind:
        tokenizer = AutoTokenizer.from_pretrained(str(cfg.teacher.model))
        llm = LLM(
            model=str(cfg.teacher.model),
            dtype=str(cfg.teacher.dtype),
            max_model_len=int(cfg.teacher.max_model_len),
            gpu_memory_utilization=float(cfg.teacher.gpu_memory_utilization),
            seed=int(cfg.seed),
        )
        prompts = [
            render_chat_prompt(tokenizer, row["blind_prompt"])
            for row in missing_blind
        ]
        completions = generate_texts(
            llm, prompts, sampling, cfg.teacher.batch_size
        )
        for row, raw_completion in zip(missing_blind, completions, strict=True):
            parsed = parse_blind_audit(
                raw_completion,
                output_from_student_prompt(row["student_prompt"]),
            )
            blind_rows.append({
                **row,
                **parsed,
                "raw_completion": raw_completion,
                "frozen_audit": format_frozen_audit(parsed),
            })
    order = {(row["dataset"], row["index"]): index for index, row in enumerate(rows)}
    blind_rows.sort(key=lambda row: order[(row["dataset"], row["index"])])
    write_jsonl(blind_path, blind_rows)
    grounded = sum(len(row["claims"]) for row in blind_rows)
    dropped = sum(len(row["ungrounded_claims"]) for row in blind_rows)
    valid = sum(bool(row["structure_valid"]) for row in blind_rows)
    print(
        f"blind valid={valid}/{len(blind_rows)} grounded_claims={grounded} "
        f"dropped_ungrounded={dropped}; wrote {blind_path}",
        flush=True,
    )

    final_rows: list[dict[str, Any]] = []
    missing_final: list[dict[str, Any]] = []
    for audit in blind_rows:
        if not audit["structure_valid"]:
            final_rows.append({
                **audit,
                "reconcile_prompt": None,
                "audit_label_relation": None,
                "reasoning_summary": None,
                "prediction": None,
                "prediction_source": None,
                "student_target": None,
                "parse_error": True,
                "label_match": False,
                "harmony_final": "",
                "reconcile_raw_completion": "",
            })
            continue
        reconcile_prompt = build_reconcile_prompt(
            audit["student_prompt"],
            audit["frozen_audit"],
            int(audit["label"]),
            str(cfg.atomic_audit.reconcile_prompt),
        )
        cached = final_cache.get((audit["dataset"], audit["index"]))
        if (
            cached
            and cached.get("student_prompt") == audit["student_prompt"]
            and cached.get("frozen_audit") == audit["frozen_audit"]
            and cached.get("reconcile_prompt") == reconcile_prompt
            and not cached.get("parse_error", True)
            and cached.get("label_match") is True
        ):
            final_rows.append(cached)
        else:
            missing_final.append({**audit, "reconcile_prompt": reconcile_prompt})
    print(
        f"reconcile cache hits={len(final_rows)} generation required={len(missing_final)}",
        flush=True,
    )

    if missing_final:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(cfg.teacher.model))
        if llm is None:
            llm = LLM(
                model=str(cfg.teacher.model),
                dtype=str(cfg.teacher.dtype),
                max_model_len=int(cfg.teacher.max_model_len),
                gpu_memory_utilization=float(cfg.teacher.gpu_memory_utilization),
                seed=int(cfg.seed),
            )
        prompts = [
            render_chat_prompt(tokenizer, row["reconcile_prompt"])
            for row in missing_final
        ]
        completions = generate_texts(
            llm, prompts, sampling, cfg.teacher.batch_size
        )
        for row, raw_completion in zip(missing_final, completions, strict=True):
            parsed = parse_reconciliation(raw_completion, int(row["label"]))
            prediction = parsed["prediction"] if parsed else None
            student_target = None
            if parsed is not None:
                student_target = format_atomic_student_target(
                    row["frozen_audit"],
                    parsed["audit_label_relation"],
                    parsed["reasoning_summary"],
                    prediction,
                )
            final_rows.append({
                **row,
                "audit_label_relation": (
                    parsed["audit_label_relation"] if parsed else None
                ),
                "reasoning_summary": parsed["reasoning_summary"] if parsed else None,
                "prediction": prediction,
                "prediction_source": parsed["prediction_source"] if parsed else None,
                "student_target": student_target,
                "parse_error": parsed is None,
                "label_match": prediction == row["label"],
                "harmony_final": parsed["harmony_final"] if parsed else "",
                "reconcile_raw_completion": raw_completion,
            })
    final_rows.sort(key=lambda row: order[(row["dataset"], row["index"])])
    write_jsonl(final_path, final_rows)
    parsed_count = sum(not row["parse_error"] for row in final_rows)
    matched = sum(row["label_match"] for row in final_rows)
    relations = {
        name: sum(row.get("audit_label_relation") == name for row in final_rows)
        for name in ("supports", "conflicts", "insufficient")
    }
    print(
        f"reconciled={parsed_count}/{len(final_rows)} label_matches={matched}/"
        f"{len(final_rows)} relations={relations}; wrote {final_path}",
        flush=True,
    )
    (final_path.parent / "config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )


if __name__ == "__main__":
    main()
