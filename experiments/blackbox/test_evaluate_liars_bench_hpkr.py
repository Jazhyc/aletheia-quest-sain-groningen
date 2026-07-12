from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.evaluate_liars_bench_hpkr import parse_prediction


def test_parse_final_binary_prediction() -> None:
    assert parse_prediction("<reasoning_summary>x</reasoning_summary>\nPrediction:1") == 1
    assert parse_prediction("Prediction:1\nCorrection Prediction:0") == 0
    assert parse_prediction("The response is deceptive") is None
