import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "submission" / "phoenix_wright_v2_4.ipynb"


def module_string_constant(path: Path, name: str) -> str:
    for statement in ast.parse(path.read_text()).body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            continue
        value = ast.literal_eval(statement.value)
        if isinstance(value, str):
            return value
    raise AssertionError(f"missing string constant {name} in {path}")


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


def test_notebook_prompt_matches_frozen_conditional_trace_config() -> None:
    constants, source = notebook_string_constants()
    reasoning_config = yaml.safe_load(
        (
            ROOT
            / "configs/privileged_information_distillation_reasoning_traces.yaml"
        ).read_text()
    )["student"]

    ordinary_prompt = constants["JUDGE_BASE_PROMPT"] + "\n\n" + constants["JUDGE_SUMMARY_CONTRACT"]
    trace_prompt = (
        constants["JUDGE_BASE_PROMPT"]
        + "\n\n"
        + constants["JUDGE_TRACE_CLAUSE"]
        + "\n\n"
        + constants["JUDGE_SUMMARY_CONTRACT"]
    )
    assert ordinary_prompt == reasoning_config["prompt_without_reasoning"]
    assert trace_prompt == reasoning_config["prompt"]
    assert "<assistant_reasoning>" in source
    assert 'PHOENIX_REASONING_MAX_CHARS", "4000"' in source
    assert "JUDGE_REASONING_PROMPT if INCLUDE_REASONING and reasoning" in source
    assert "JUDGE_BINARY" not in source
    assert "BINARY_RECALL_MEMBER" not in source
    assert "np.maximum" not in source


def test_notebook_action_route_matches_frozen_external_prompt() -> None:
    constants, source = notebook_string_constants()
    action_prompt = module_string_constant(
        ROOT / "experiments/liars_bench_distillation/prompts.py",
        "ACTION_REPORT_PROMPT",
    )

    assert constants["JUDGE_ACTION_PROMPT"] == action_prompt
    assert 'PHOENIX_ACTION_MAX_PROMPT_CHARS", "4500"' in source
    assert "is_action_report_conversation(messages)" in source
    assert "truncate_head_tail(raw_context, ACTION_MAX_PROMPT_CHARS)" in source
    assert "full_final_assistant(messages) if action_report" in source
    assert "elif action_report:" in source


def test_notebook_passage_route_matches_frozen_external_prompt() -> None:
    constants, source = notebook_string_constants()
    passage_prompt = module_string_constant(
        ROOT / "experiments/liars_bench_distillation/prompts.py",
        "PASSAGE_TRUE_FALSE_PROMPT",
    )

    assert constants["JUDGE_PASSAGE_PROMPT"] == passage_prompt
    assert 'PHOENIX_PASSAGE_MAX_PROMPT_CHARS", "6000"' in source
    assert "is_passage_true_false_request(messages)" in source
    assert "truncate_head_tail(raw_context, PASSAGE_MAX_PROMPT_CHARS)" in source
    assert "elif passage_true_false:" in source


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
