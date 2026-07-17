import json

from experiments.liars_bench_distillation.prepare_heavy_spectrum_confirmation import (
    exclusion_ids,
)


def test_exclusion_ids_normalizes_raw_and_prefixed_indices(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join([
        json.dumps({"dataset": "liars-bench/soft-trigger", "index": "soft-trigger:7"}),
        json.dumps({"category": "insider-trading", "index": 9}),
    ]) + "\n")
    assert exclusion_ids([path]) == {"soft-trigger:7", "insider-trading:9"}
