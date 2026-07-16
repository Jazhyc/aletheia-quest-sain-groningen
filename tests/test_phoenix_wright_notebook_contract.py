import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "submission" / "phoenix_wright_v2_3.ipynb"


def notebook_string_constants() -> tuple[dict[str, str], str]:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    constants = {}
    for statement in ast.parse(source).body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, str):
            constants[target.id] = value
    return constants, source


def test_notebook_prompts_match_frozen_evaluator_configs() -> None:
    constants, _ = notebook_string_constants()
    base = constants["JUDGE_BASE_PROMPT"]
    trace = constants["JUDGE_TRACE_CLAUSE"]
    summary = constants["JUDGE_SUMMARY_CONTRACT"]
    binary = constants["JUDGE_BINARY_CONTRACT"]

    ordinary_config = yaml.safe_load(
        (ROOT / "configs/privileged_information_distillation.yaml").read_text()
    )["student"]
    trace_config = yaml.safe_load(
        (
            ROOT
            / "configs/privileged_information_distillation_reasoning_traces.yaml"
        ).read_text()
    )["student"]
    binary_config = yaml.safe_load(
        (
            ROOT
            / "configs/privileged_information_distillation_reasoning_binary4000.yaml"
        ).read_text()
    )["student"]

    assert base + "\n\n" + summary == ordinary_config["prompt"]
    assert base + "\n\n" + trace + "\n\n" + summary == trace_config["prompt"]
    assert base + "\n\n" + binary == binary_config["prompt_without_reasoning"]
    assert base + "\n\n" + trace + "\n\n" + binary == binary_config["prompt"]


def test_notebook_uses_language_model_and_safe_adaptive_defaults() -> None:
    _, source = notebook_string_constants()

    assert 'PHOENIX_MODEL_WRAPPER", "language"' in source
    assert 'PHOENIX_BATCH_SIZE", "32"' in source
    assert 'PHOENIX_MEDIUM_PROMPT_BATCH_SIZE", "24"' in source
    assert 'PHOENIX_LONG_PROMPT_BATCH_SIZE", "16"' in source
    assert 'PHOENIX_REMOTE_BATCHES_PER_SESSION", "2"' in source
    assert "verdict_explanation_conflict" not in source
    assert "consistency_scores" not in source
