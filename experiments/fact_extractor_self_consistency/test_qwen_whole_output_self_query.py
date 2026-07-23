from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fact_extractor_self_consistency.run_qwen_whole_output_self_query import (
    parse_boolean,
)


def test_parse_boolean_accepts_strict_verdicts() -> None:
    assert parse_boolean("TRUE") is True
    assert parse_boolean(" false. \n") is False


def test_parse_boolean_rejects_explanations() -> None:
    assert parse_boolean("The answer is TRUE") is None
    assert parse_boolean("TRUE because all claims check out") is None
