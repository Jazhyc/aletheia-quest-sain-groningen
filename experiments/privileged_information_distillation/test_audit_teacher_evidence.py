from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.audit_teacher_evidence import (
    audit_teacher_caches,
)


def row(summary: str, *, teacher_only: bool = True) -> dict:
    return {
        "dataset": "dataset",
        "index": 1,
        "label": 1,
        "student_prompt": "PROMPT" if teacher_only else "PROMPT <reference_material>",
        "reasoning_summary": summary,
        "parse_error": False,
        "label_match": True,
    }


def test_audit_teacher_caches_tracks_source_specific_changes() -> None:
    report = audit_teacher_caches(
        [row("The date is wrong.")],
        [row("The 1997 date is wrong.")],
        [row("The unrelated detail is wrong.")],
        [{
            "dataset": "dataset",
            "index": 1,
            "real_passages": [{"title": "Event", "text": "It happened in 1997."}],
            "shuffled_passages": [{"title": "Other", "text": "An unrelated detail."}],
        }],
    )

    assert report["counts"]["active_rows"] == 1
    assert report["counts"]["real_changed_from_baseline"] == 1
    assert report["counts"]["real_introduces_source_term"] == 1
    assert report["counts"]["student_reference_leaks"] == 0
