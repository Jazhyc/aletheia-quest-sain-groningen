# Black-Box Experiment Leaderboard

Timing is scoring-only wall time: it excludes vLLM startup/model load/compile and dataset preparation.
Submitted timestamps are UTC.

| submitted_at | method | AUROC | bal_acc | recall | FPR | score_time | rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-03 01:50:21 | qwen_reason_v1 | 0.9092 | 0.9095 | 0.8357 | 0.0167 | 428.2s | 1.9/s |
| 2026-07-03 00:21:22 | qwen_judge_v1 | 0.8657 | 0.6393 | 0.3048 | 0.0262 | 12.4s | 66.4/s |
