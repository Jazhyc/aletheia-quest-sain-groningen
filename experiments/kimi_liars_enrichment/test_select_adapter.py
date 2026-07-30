from experiments.kimi_liars_enrichment.select_adapter import choose


ANCHOR = {
    "macro_auroc": 0.964,
    "instructed_auroc": 0.998,
    "varied_auroc": 0.918,
}


def test_selection_uses_half_for_close_validation_and_liars_tie() -> None:
    candidates = {
        "full": {
            "macro_auroc": 0.9644,
            "instructed_auroc": 0.998,
            "varied_auroc": 0.9184,
        },
        "half": {
            "macro_auroc": 0.9641,
            "instructed_auroc": 0.998,
            "varied_auroc": 0.9181,
        },
    }
    result = choose(
        ANCHOR,
        candidates,
        {"anchor": 0.80, "full": 0.85, "half": 0.849},
    )
    assert result["selected"] == "half"


def test_selection_rejects_liars_gain_that_regresses_competition() -> None:
    result = choose(
        ANCHOR,
        {
            "full": {
                "macro_auroc": 0.960,
                "instructed_auroc": 0.998,
                "varied_auroc": 0.910,
            }
        },
        {"anchor": 0.80, "full": 0.95},
    )
    assert result["selected"] is None
    assert not result["candidates"]["full"]["gates"]["competition_macro"]
