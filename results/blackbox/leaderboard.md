# Black-Box Experiment Leaderboard

Timing is scoring-only wall time: it excludes vLLM startup/model load/compile and dataset preparation.
Submitted timestamps are UTC.

| submitted_at | method | AUROC | bal_acc | recall | FPR | score_time | rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-03 22:37:17 | qwen_reason_ensemble_dks_member4096_v1 | 0.9340 | 0.9298 | 0.8976 | 0.0381 | 1284.6s | 0.6/s |
| 2026-07-03 17:17:24 | qwen_reason_budget_8192 | 0.8915 | 0.8917 | 0.8024 | 0.0190 | 645.9s | 1.3/s |
| 2026-07-03 17:01:08 | qwen_reason_budget_2048 | 0.8679 | 0.8679 | 0.7548 | 0.0190 | 278.2s | 3.0/s |
| 2026-07-03 14:21:30 | qwen_structured_reason_budget_2048 | 0.9185 | 0.8762 | 0.7667 | 0.0143 | 384.4s | 2.1/s |
| 2026-07-03 01:50:21 | qwen_reason_v1 | 0.9092 | 0.9095 | 0.8357 | 0.0167 | 428.2s | 1.9/s |
| 2026-07-03 00:21:22 | qwen_judge_v1 | 0.8657 | 0.6393 | 0.3048 | 0.0262 | 12.4s | 66.4/s |
