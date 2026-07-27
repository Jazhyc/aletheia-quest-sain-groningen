from pathlib import Path

import pandas as pd

from experiments.persona_belief_prompt_sweep.analyze_pid_matched import (
    condition_result,
    load_condition_frame,
)


def test_condition_result_uses_adapter_parent_and_named_condition():
    adapter = Path("/tmp/method/adapter")
    assert condition_result(adapter, "validation_run", "persona_prompt") == Path(
        "/tmp/method/validation_run/persona_prompt/result.json"
    )


def test_load_condition_frame_aligns_stringified_indices(tmp_path):
    condition_dir = tmp_path / "condition"
    condition_dir.mkdir()
    (condition_dir / "result.json").write_text("{}\n")
    pd.DataFrame({
        "dataset": ["dataset-a", "dataset-a"],
        "index": [10, 20],
        "direct_margin_score": [0.25, 0.75],
    }).to_json(
        condition_dir / "generations.jsonl",
        orient="records",
        lines=True,
    )
    templates = [pd.DataFrame({
        "_key": ["20", "10"],
        "label": [1, 0],
        "dataset": ["dataset-a", "dataset-a"],
        "persona": [True, False],
    })]

    frames = load_condition_frame(
        condition_dir / "result.json",
        templates,
    )

    assert frames[0]["score"].tolist() == [0.75, 0.25]
