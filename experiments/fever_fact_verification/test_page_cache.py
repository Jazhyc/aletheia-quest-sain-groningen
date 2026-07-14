from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fever_fact_verification.build_from_page_cache import select_claim_passages
from experiments.fever_fact_verification.fetch_page_cache import resolve_requested_pages


def test_resolve_requested_redirect() -> None:
    payload = {
        "query": {
            "redirects": [{"from": "NYC", "to": "New York City"}],
            "pages": [{
                "title": "New York City",
                "fullurl": "https://example.test/wiki/New_York_City",
                "extract": "New York City is in the United States.",
            }],
        }
    }
    row = resolve_requested_pages(["NYC"], payload)[0]
    assert row["canonical_title"] == "New York City"
    assert not row["missing"]


def test_resolve_empty_extract_is_retryable_error() -> None:
    payload = {"query": {"pages": [{"title": "Empty", "extract": ""}]}}
    row = resolve_requested_pages(["Empty"], payload)[0]
    assert row["error"] == "empty extract"


def test_select_claim_passages_uses_buried_exact_date() -> None:
    claim = {
        "proposition": "The organization was founded in 1974.",
        "passages": [{"title": "Example", "page_rank": 0}],
    }
    cache = {
        "Example": {
            "canonical_title": "Example",
            "url": "https://example.test",
            "extract": (
                "The organization has a long history. "
                "It is based in Europe and has many members. "
                "A different organization began in 1999. "
                "The organization was founded in 1974."
            ),
            "error": None,
        }
    }
    passages, errors = select_claim_passages(claim, cache, sentences_per_page=3)
    assert not errors
    assert any("founded in 1974" in passage["text"] for passage in passages)


def test_select_claim_passages_ignores_unrequested_page_ranks() -> None:
    claim = {
        "proposition": "A claim.",
        "passages": [
            {"title": "First", "page_rank": 0},
            {"title": "Second", "page_rank": 1},
        ],
    }
    cache = {
        "First": {
            "canonical_title": "First",
            "url": "",
            "extract": "This is a sufficiently long first-page sentence.",
            "error": None,
        }
    }
    _, errors = select_claim_passages(
        claim, cache, sentences_per_page=3, max_page_rank=0
    )
    assert not errors
