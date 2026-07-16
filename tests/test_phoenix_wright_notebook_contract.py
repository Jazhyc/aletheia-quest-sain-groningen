import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "submission" / "phoenix_wright_v2_1.ipynb"


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


def test_notebook_prompt_matches_frozen_summary_only_config() -> None:
    constants, source = notebook_string_constants()
    ordinary_config = yaml.safe_load(
        (ROOT / "configs/privileged_information_distillation.yaml").read_text()
    )["student"]

    assert constants["JUDGE_SYSTEM_PROMPT"] == ordinary_config["prompt"]
    assert "<assistant_reasoning>" not in source
    assert "JUDGE_BINARY" not in source
    assert "BINARY_RECALL_MEMBER" not in source
    assert "np.maximum" not in source


def test_notebook_uses_language_model_and_safe_adaptive_defaults() -> None:
    _, source = notebook_string_constants()

    assert "from nnsight import LanguageModel" in source
    assert "from nnsight.modeling.language import LanguageModel" in source
    assert "VisionLanguageModel" not in source
    assert 'PHOENIX_BATCH_SIZE", "32"' in source
    assert 'PHOENIX_MEDIUM_PROMPT_BATCH_SIZE", "24"' in source
    assert 'PHOENIX_LONG_PROMPT_BATCH_SIZE", "16"' in source
    assert 'PHOENIX_REMOTE_BATCHES_PER_SESSION", "2"' in source
    assert "verdict_explanation_conflict" not in source
    assert "consistency_scores" not in source
