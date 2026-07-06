# Black-Box Experiment Leaderboard

Timing is scoring-only wall time: it excludes vLLM startup/model load/compile and dataset preparation.
Submitted timestamps are UTC.
Rows with cached text-probe components report the logits scoring wall time; CPU n-gram inference is negligible.

| submitted_at | method | AUROC | bal_acc | recall | FPR | score_time | rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-06 07:51:47 | qwen_grpo_lora_r16_reasonfield_muonlr3e5_full_v1 | 0.8988 | 0.8988 | 0.8667 | 0.0690 | 101.7s | 8.1/s |
| 2026-07-05 18:18:35 | qwen_reason_nothink_reasoning_hierarchy_sweep_v23 | 0.8869 | 0.8869 | 0.8476 | 0.0738 | 102.1s | 8.0/s |
| 2026-07-05 18:06:15 | qwen_reason_nothink_reasoning_output_consistency_v1 | 0.8762 | 0.8762 | 0.8214 | 0.0690 | 19.9s | 41.3/s |
| 2026-07-05 15:50:25 | qwen_reason_nothink_truth_hidden_ensemble_v1 | 0.8818 | 0.8821 | 0.8357 | 0.0714 | 29.7s | 27.7/s |
| 2026-07-05 14:55:24 | qwen_reason_details_nothink_t512_t01_v1 | 0.8570 | 0.8571 | 0.7643 | 0.0500 | 12.2s | 67.4/s |
| 2026-07-04 14:11:38 | phoenix_wright_v1_2_outcome_rank067_ngram | 0.9160 | 0.8786 | 0.9333 | 0.1762 | 46.9s | 17.5/s |
| 2026-07-04 14:11:38 | qwen_reason_ensemble_dk3072_logit_outcome_v1_test | 0.8952 | 0.6762 | 0.3690 | 0.0167 | 46.9s | 17.5/s |
| 2026-07-04 13:04:28 | phoenix_wright_v1_1_rankavg_dk_logits_ngram | 0.9068 | 0.8655 | 0.8071 | 0.0762 | 24.1s | 34.1/s |
| 2026-07-04 13:04:28 | qwen_reason_ensemble_dk3072_logit_v1 | 0.8804 | 0.8369 | 0.6976 | 0.0238 | 24.1s | 34.1/s |
| 2026-07-03 22:37:17 | qwen_reason_ensemble_dks_member4096_v1 | 0.9340 | 0.9298 | 0.8976 | 0.0381 | 1284.6s | 0.6/s |
| 2026-07-03 17:17:24 | qwen_reason_budget_8192 | 0.8915 | 0.8917 | 0.8024 | 0.0190 | 645.9s | 1.3/s |
| 2026-07-03 17:01:08 | qwen_reason_budget_2048 | 0.8679 | 0.8679 | 0.7548 | 0.0190 | 278.2s | 3.0/s |
| 2026-07-03 14:21:30 | qwen_structured_reason_budget_2048 | 0.9185 | 0.8762 | 0.7667 | 0.0143 | 384.4s | 2.1/s |
| 2026-07-03 01:50:21 | qwen_reason_v1 | 0.9092 | 0.9095 | 0.8357 | 0.0167 | 428.2s | 1.9/s |
| 2026-07-03 00:21:22 | qwen_judge_v1 | 0.8657 | 0.6393 | 0.3048 | 0.0262 | 12.4s | 66.4/s |
