from __future__ import annotations

from experiments.privileged_information_distillation.build_qwen27_dks_retry_split import (
    failed_row_keys,
)


def test_failed_row_keys_requires_all_closed_members() -> None:
    base = [
        {
            "dataset": "dev-varied-deception-a",
            "index": index,
            "parse_error": False,
            "label_match": True,
        }
        for index in (1, 2)
    ]
    generations = [
        {
            "dataset": "dev-varied-deception-a",
            "index": index,
            "ensemble_member": member,
            "parse_error": index == 2 and member == "known4096",
            "text": "<think>x</think>\nRating: 1",
        }
        for index in (1, 2)
        for member in ("details4096", "known4096", "scrutiny4096")
    ]

    assert failed_row_keys(base, generations) == {
        ("dev-varied-deception-a", 2)
    }
