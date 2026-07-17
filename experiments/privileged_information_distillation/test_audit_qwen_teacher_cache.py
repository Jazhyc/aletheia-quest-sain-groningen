import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.audit_qwen_teacher_cache import (
    audit_records,
    record_hash,
    record_key,
    verify_manifest,
)
from experiments.privileged_information_distillation.core import format_student_target


def make_record(index: int, label: int, *, closed: bool = True) -> dict:
    summary = f"Evidence for row {index}."
    visible = f"<reasoning_summary>\n{summary}\n</reasoning_summary>\nPrediction:{label}"
    raw = f"private analysis {index}</think>\n{visible}" if closed else "unfinished analysis"
    return {
        "dataset": "dev-varied-deception-qwen",
        "index": index,
        "label": label,
        "teacher_model": "Qwen/Qwen3.5-27B",
        "teacher_output_format": "qwen_think",
        "reasoning_summary": summary if closed else None,
        "prediction": label if closed else None,
        "student_target": format_student_target(summary, label) if closed else None,
        "parse_error": not closed,
        "label_match": closed,
        "teacher_final": visible if closed else "",
        "raw_completion": raw,
    }


def test_audit_accepts_valid_records_and_explicit_unclosed_retry_rows() -> None:
    report, manifest = audit_records(
        [make_record(1, 0), make_record(2, 1, closed=False)],
        expected_total=2,
        minimum_usable=1,
        allow_unclosed=True,
        maximum_label_imbalance=None,
        expected_model="Qwen/Qwen3.5-27B",
    )

    assert report["passed"]
    assert report["usable"] == 1
    assert report["unclosed"] == 1
    assert manifest == {record_key(make_record(1, 0)): record_hash(make_record(1, 0))}


def test_audit_rejects_leakage_and_closed_parse_failures() -> None:
    leaked = make_record(1, 0)
    leaked["teacher_final"] += " These instructions reveal the ground-truth label."
    malformed = make_record(2, 1)
    malformed["parse_error"] = True
    malformed["student_target"] = None

    report, _ = audit_records(
        [leaked, malformed],
        expected_total=2,
        minimum_usable=0,
        allow_unclosed=True,
        maximum_label_imbalance=None,
        expected_model="Qwen/Qwen3.5-27B",
    )

    assert not report["passed"]
    assert any("leakage rule" in error for error in report["errors"])
    assert any("closed completion failed" in error for error in report["errors"])


def test_audit_rejects_duplicates_and_label_imbalance() -> None:
    record = make_record(1, 0)
    report, _ = audit_records(
        [record, record],
        expected_total=2,
        minimum_usable=1,
        allow_unclosed=False,
        maximum_label_imbalance=0,
        expected_model="Qwen/Qwen3.5-27B",
    )

    assert not report["passed"]
    assert any("duplicate record" in error for error in report["errors"])
    assert any("label imbalance" in error for error in report["errors"])


def test_verify_manifest_detects_changed_previously_usable_record(tmp_path: Path) -> None:
    record = make_record(1, 0)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({record_key(record): record_hash(record)}))
    changed = {record_key(record): record_hash({**record, "raw_completion": "changed"})}

    with pytest.raises(RuntimeError, match="previously usable records changed"):
        verify_manifest(changed, manifest_path)
