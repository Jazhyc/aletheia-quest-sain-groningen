from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.audit_qwen_organism_claim_consistency import (
    parse_boolean,
    parse_claims,
)


def test_parse_claims_deduplicates_and_normalizes() -> None:
    text = "<claim>A  fact.</claim>\n<claim>A fact.</claim>\n<claim>B fact.</claim>"
    assert parse_claims(text) == ["A fact.", "B fact."]


def test_parse_boolean_requires_explicit_word() -> None:
    assert parse_boolean("False") is False
    assert parse_boolean("True") is True
    assert parse_boolean("uncertain") is None
