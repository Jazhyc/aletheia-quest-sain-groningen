from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privileged_information_distillation.build_wikipedia_rag_cache import (
    reusable_queries,
)


def test_reusable_queries_keeps_successful_searches_only() -> None:
    records = [
        {"query": "same", "passages": [{"title": "A", "text": "B"}], "error": None},
        {"query": "failed", "passages": [], "error": "timeout"},
        {"query": "", "passages": [], "error": None},
    ]

    assert reusable_queries(records) == {
        "same": [{"title": "A", "text": "B"}]
    }
