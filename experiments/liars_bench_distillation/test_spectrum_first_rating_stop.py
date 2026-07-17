from experiments.liars_bench_distillation.analyze_spectrum_first_rating_stop import (
    FIRST_RATING_RE,
)


def test_first_rating_pattern_requires_complete_valid_rating() -> None:
    assert FIRST_RATING_RE.search("analysis\nRating: 7\nmore")
    assert not FIRST_RATING_RE.search("Rating: 8")
    assert not FIRST_RATING_RE.search("Rating: ")
