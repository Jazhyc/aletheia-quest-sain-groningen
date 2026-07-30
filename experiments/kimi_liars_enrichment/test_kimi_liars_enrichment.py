from experiments.kimi_liars_enrichment.prompts import CATEGORY_SPECS
from experiments.kimi_liars_enrichment.query import make_work


def row(category: str, label: int = 0) -> dict:
    return {
        "dataset": f"liars-bench/{category}",
        "index": f"{category}:1",
        "category": category,
        "label": label,
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    }


def test_declared_full_mixture_has_6576_rows_and_balanced_cells():
    assert sum(2 * int(spec["train_per_label"]) for spec in CATEGORY_SPECS.values()) == 6576


def test_semantic_work_uses_category_prompt():
    work = make_work([row("harm-pressure-knowledge-report", 1)], "semantic")
    assert "POLICY_REFUSAL" in work[0].prompt
    assert work[0].label == 1


def test_ordinary_work_uses_general_truth_value_guard():
    work = make_work([row("harm-pressure-choice")], "ordinary")
    assert "factual answers" in work[0].prompt
    assert "forced-choice factual answers" not in work[0].prompt


def test_selected_work_uses_row_condition():
    selected = row("insider-trading")
    selected["teacher_prompt_kind"] = "ordinary"
    work = make_work([selected], "selected")
    assert "reports about actions" not in work[0].prompt
