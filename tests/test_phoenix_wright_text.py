import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))

from phoenix_wright_text import safe_text


def test_safe_text_replaces_lone_surrogates():
    text = safe_text("before " + chr(0xD800) + " after " + chr(0xDC00))

    assert "\ud800" not in text
    assert "\udc00" not in text
    assert "before ? after ?" == text


def test_safe_text_accepts_non_string_values():
    assert safe_text(None) == "None"
    assert safe_text(123) == "123"
    assert safe_text({"why": "because"}) == "{'why': 'because'}"
