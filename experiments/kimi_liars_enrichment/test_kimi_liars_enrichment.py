from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.kimi_liars_enrichment.prompts import CATEGORY_SPECS
from experiments.kimi_liars_enrichment.query import make_work


ROOT = Path(__file__).resolve().parents[2]


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


def load_config(name: str) -> dict:
    with initialize_config_dir(
        config_dir=(ROOT / "configs").as_posix(), version_base=None
    ):
        return OmegaConf.to_container(
            compose(config_name=name), resolve=True
        )


def test_full_and_half_training_configs_use_combined_soft_cache():
    full = load_config("pid_kimi_k3_liars_binary_soft_full_r16_ep2_v1")
    half = load_config("pid_kimi_k3_liars_binary_soft_half_r16_ep2_v1")
    expected = (
        "results/blackbox/kimi_k3_tvg_soft_full_plus_liars_v1/"
        "train/soft_targets.jsonl"
    )

    assert full["student"]["soft_teacher_artifact"] == expected
    assert half["student"]["soft_teacher_artifact"] == expected
    assert full["student"]["training"]["learning_rate"] == 5e-5
    assert full["student"]["training"]["num_train_epochs"] == 2.0
    assert half["student"]["teacher_sources"] == [
        {
            "artifact": (
                "results/blackbox/kimi_k3_tvg_soft_full_plus_liars_v1/"
                "train/student_rows.jsonl"
            ),
            "dataset_name_contains": "aletheias-quest/",
        },
        {
            "artifact": (
                "results/blackbox/kimi_k3_tvg_soft_full_plus_liars_v1/"
                "train/student_rows.jsonl"
            ),
            "dataset_name_contains": "liars-bench/",
            "train_fraction": 0.5,
            "train_fraction_seed": 0,
        },
    ]
