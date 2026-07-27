# Black-Box Experiment Leaderboard

Timing is scoring-only wall time: it excludes vLLM startup/model load/compile and dataset preparation.
Submitted timestamps are UTC.

| submitted_at | method | AUROC | bal_acc | recall | FPR | score_time | rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-27 15:24:16 | qwen9b_qwen27_tvg_binary_softonly_varied_v1 | 0.9415 | 0.8679 | 0.8143 | 0.0786 | 13.0s | 63.3/s |
| 2026-07-27 15:06:02 | qwen35_122b_openrouter_nothink_tvg_binary_logit_v1 | 0.9555 | 0.8952 | 0.8333 | 0.0429 | 164.7s | 5.0/s |
| 2026-07-27 15:04:22 | qwen35_397b_openrouter_nothink_tvg_binary_logit_v1 | 0.9654 | 0.9071 | 0.8500 | 0.0357 | 125.1s | 6.6/s |
| 2026-07-27 13:00:07 | qwen9b_pid_varied_grpo_ep1_v1_logits_empty_reasoning_plain | 0.9598 | 0.8869 | 0.8286 | 0.0548 | 38.9s | 21.1/s |
| 2026-07-25 23:45:22 | qwen9b_reasoning_intent_logits_v1 | 0.8505 | 0.6202 | 0.2524 | 0.0119 | 30.7s | 26.8/s |
| 2026-07-17 02:02:26 | qwen9b_reason_ensemble_dks_first_rating_stop_v1 | 0.9332 | 0.9310 | 0.8881 | 0.0262 | 528.3s | 1.6/s |
| 2026-07-17 00:23:59 | qwen27b_reason_ensemble_dks_member4096_v1 | 0.9573 | 0.9464 | 0.9357 | 0.0429 | 1484.5s | 0.6/s |
| 2026-07-16 08:54:51 | continuous_hybrid_dks3072_logits_v1 | 0.9014 | 0.8726 | 0.7857 | 0.0405 | 77.7s | 10.6/s |
| 2026-07-11 16:32:24 | qwen9b_pid_varied_grpo_ep2_v1 | 0.9048 | 0.9048 | 0.8310 | 0.0214 | 147.4s | 5.6/s |
| 2026-07-11 15:55:23 | gpt_oss_120b_reason_ensemble_dks_member4096_v1 | 0.9465 | 0.9345 | 0.9000 | 0.0310 | 104.0s | 7.9/s |
| 2026-07-11 03:35:07 | gpt_oss_120b_nothink_truth_value_v1 | 0.9391 | 0.9310 | 0.8976 | 0.0357 | 25.2s | 32.6/s |
| 2026-07-08 09:21:27 | qwen_reason_nothink_truth_value_pc1500_v1 | 0.8761 | 0.8762 | 0.8238 | 0.0714 | 13.6s | 60.4/s |
| 2026-07-07 16:16:29 | qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2_logits_empty_reasoning_space_plain | 0.8686 | 0.8810 | 0.7810 | 0.0190 | 60.4s | 13.6/s |
| 2026-07-07 15:47:50 | qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2_logits_empty_reasoning_space_plain | 0.9646 | 0.9024 | 0.8714 | 0.0667 | 51.0s | 16.1/s |
| 2026-07-07 15:46:23 | qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2_logits_empty_reasoning_plain | 0.9654 | 0.9048 | 0.8690 | 0.0595 | 55.3s | 14.9/s |
| 2026-07-06 14:08:07 | qwen_grpo_lora_r16_binary_t12_muonlr3e5_full_v2 | 0.8762 | 0.8762 | 0.7714 | 0.0190 | 40.7s | 20.2/s |
| 2026-07-06 08:15:36 | qwen_grpo_lora_r16_reasonfield_muonlr3e5_full_v1 | 0.8758 | 0.8738 | 0.7762 | 0.0286 | 38.5s | 21.3/s |
| 2026-07-05 18:18:35 | qwen_reason_nothink_reasoning_hierarchy_sweep_v23 | 0.8869 | 0.8869 | 0.8476 | 0.0738 | 102.1s | 8.0/s |
| 2026-07-05 18:06:15 | qwen_reason_nothink_reasoning_output_consistency_v1 | 0.8762 | 0.8762 | 0.8214 | 0.0690 | 19.9s | 41.3/s |
| 2026-07-05 15:50:25 | qwen_reason_nothink_truth_hidden_ensemble_v1 | 0.8818 | 0.8821 | 0.8357 | 0.0714 | 29.7s | 27.7/s |
| 2026-07-05 14:55:24 | qwen_reason_details_nothink_t512_t01_v1 | 0.8570 | 0.8571 | 0.7643 | 0.0500 | 12.2s | 67.4/s |
| 2026-07-04 14:11:38 | qwen_reason_ensemble_dk3072_logit_outcome_v1_test | 0.8952 | 0.6762 | 0.3690 | 0.0167 | 46.9s | 17.5/s |
| 2026-07-04 13:04:28 | qwen_reason_ensemble_dk3072_logit_v1 | 0.8804 | 0.8369 | 0.6976 | 0.0238 | 24.1s | 34.1/s |
| 2026-07-03 22:37:17 | qwen_reason_ensemble_dks_member4096_v1 | 0.9340 | 0.9298 | 0.8976 | 0.0381 | 1284.6s | 0.6/s |
| 2026-07-03 17:17:24 | qwen_reason_budget_8192 | 0.8915 | 0.8917 | 0.8024 | 0.0190 | 645.9s | 1.3/s |
| 2026-07-03 17:01:08 | qwen_reason_budget_2048 | 0.8679 | 0.8679 | 0.7548 | 0.0190 | 278.2s | 3.0/s |
| 2026-07-03 14:21:30 | qwen_structured_reason_budget_2048 | 0.9185 | 0.8762 | 0.7667 | 0.0143 | 384.4s | 2.1/s |
| 2026-07-03 01:50:21 | qwen_reason_v1 | 0.9092 | 0.9095 | 0.8357 | 0.0167 | 428.2s | 1.9/s |
| 2026-07-03 00:21:22 | qwen_judge_v1 | 0.8657 | 0.6393 | 0.3048 | 0.0262 | 12.4s | 66.4/s |
| - | qwen9b_pid_varied_rank24_full_adamw5e5_v1 | - | - | - | - | - | - |
| - | qwen9b_privileged_gptoss120b_summary_adamwlr5e5_v1 | - | - | - | - | - | - |
| - | continuous_hybrid_locked_test_v1 | 0.9571 | 0.9298 | 0.8976 | 0.0381 | - | - |
| - | qwen9b_pid_family_coverage_varied10_adamw5e5_v1 | - | - | - | - | - | - |
| - | qwen9b_pid_varied_muonlr1e4_ep2_v1 | - | - | - | - | - | - |
| - | qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1 | - | - | - | - | - | - |
| - | qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1 | - | - | - | - | - | - |
| - | qwen9b_privileged_gptoss120b_summary_variedonly_adamwlr5e5_v1 | - | - | - | - | - | - |
| - | qwen9b_heterogeneous_resolved_intent_rank1_v1 | - | - | - | - | - | - |
| - | qwen9b_heterogeneous_resolved_intent_rank1_v1 | - | - | - | - | - | - |
| - | qwen9b_pid_predictiononly_variedonly_adamwlr5e5_v1 | - | - | - | - | - | - |
| - | qwen9b_heterogeneous_incorrectness_rank1_v1 | - | - | - | - | - | - |
| - | qwen9b_pid_specialist_material_rank1_v1 | - | - | - | - | - | - |
| - | qwen9b_pid_varied_datafrac10_adamw5e5_v1 | - | - | - | - | - | - |
| - | qwen9b_pid_teacher_polarity_guard_variedonly_v1 | - | - | - | - | - | - |
| - | qwen9b_pid_reasoning_headtail2400_variedonly_adamw5e5_v1 | - | - | - | - | - | - |
