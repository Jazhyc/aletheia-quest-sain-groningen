from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.fingerprint_phoenix_adapters_ndif import (
    compare_scores,
)


def test_compare_scores_reports_exact_and_order_differences() -> None:
    result = compare_scores(
        "left",
        [0.1, 0.2, 0.3],
        "right",
        [0.1, 0.25, 0.2],
    )

    assert result["rows"] == 3
    assert result["exact_equal_rows"] == 1
    assert result["close_rows_at_1e_7"] == 1
    assert result["same_full_order"] is False
    assert abs(result["mean_absolute_difference"] - 0.05) < 1e-12
    assert abs(result["max_absolute_difference"] - 0.1) < 1e-12
