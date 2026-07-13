from experiments.wikidata_rag.build_broad_index import daily_snapshots, month_range, page_qids
from experiments.wikidata_rag.evaluate_offline_index import fts_expression, prepare_source


def test_month_range_crosses_year() -> None:
    assert month_range("2025-11", "2026-02") == [
        (2025, 11), (2025, 12), (2026, 1), (2026, 2),
    ]


def test_daily_snapshots_respect_step_across_months() -> None:
    assert daily_snapshots("2025-01", "2025-02", 20) == [
        (2025, 1, 1), (2025, 1, 21), (2025, 2, 10),
    ]


def test_page_qids_follows_normalization_and_redirects() -> None:
    data = {
        "query": {
            "normalized": [{"from": "Douglas_Adams", "to": "Douglas Adams"}],
            "redirects": [{"from": "Douglas Adams", "to": "Douglas Noël Adams"}],
            "pages": [{
                "title": "Douglas Noël Adams",
                "pageprops": {"wikibase_item": "Q42"},
            }],
        }
    }
    assert page_qids(data) == {
        "Douglas Noël Adams": "Q42",
        "Douglas_Adams": "Q42",
        "Douglas Adams": "Q42",
    }


def test_fts_expression_quotes_phrases_and_ors_keywords() -> None:
    assert fts_expression("French Foreign Legion", phrase=True) == '"french foreign legion"'
    expression = fts_expression("Who founded France?", phrase=False)
    assert set(expression.split(" OR ")) == {'"who"', '"founded"', '"france"'}


def test_prepare_source_filters_and_extracts_teacher_rows() -> None:
    rows = [
        {
            "dataset": "varied-deception-demo",
            "student_prompt": (
                "System text\nConversation:\nUser: Is this true?\n"
                "Assistant: Yes.\n\nAssess it."
            ),
            "reasoning_summary": "The answer asserts the target claim.",
            "parse_error": False,
        },
        {
            "dataset": "instructed-deception-demo",
            "conversation": "excluded",
            "reasoning_summary": "excluded",
        },
        {
            "dataset": "varied-deception-demo",
            "conversation": "excluded parse error",
            "reasoning_summary": "excluded",
            "parse_error": True,
        },
    ]

    prepared = prepare_source(rows, "varied-deception")

    assert len(prepared) == 1
    assert "User: Is this true?" in prepared[0]["conversation"]
