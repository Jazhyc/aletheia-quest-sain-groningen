# Prompt Optimization Loop

Goal: improve the Qwen reasoning black-box judge beyond the current local
validation baseline while reducing scoring time when possible.

Target:

- Validation balanced accuracy: 0.93 or higher.
- Runtime: lower than the current 4096-token regular reasoning method when that
  does not materially hurt balanced accuracy.

Guardrails:

- Tune on `validation`; reserve local `test` for confirmation only.
- Keep prompt logic general across organisms and scenarios.
- Track every proposed prompt, threshold change, and run outcome in this folder.
- Prefer validation evidence over intuition. Each candidate should report mean
  balanced accuracy, AUROC, recall, FPR, parse errors, score time, and worst
  datasets.

Loop:

1. Audit the baseline for threshold sensitivity, worst datasets, parse errors,
   and representative false negatives/false positives.
2. Propose one prompt or threshold change at a time.
3. Run validation through `experiments/blackbox/run_judge.py` or the Slurm
   wrapper.
4. Record the result in `experiment_log.md`.
5. Promote only candidates that improve balanced accuracy or preserve accuracy
   with lower time.

Useful commands:

```bash
.venv/bin/python scripts/audit_blackbox_run.py results/blackbox/qwen_reason_v1/validation/result.json
sbatch experiments/blackbox/run_judge.sh --config-name blackbox_reasoning_recall4096
sbatch experiments/blackbox/run_judge.sh --config-name blackbox_reasoning_recall2048
```

Submission mapping:

- `submission/phoenix_wright_v3_0.ipynb` is the current accuracy-first NDIF
  submission. It deploys a timeout-reduced two-member form of P67's rank-1
  heterogeneous stack: material deception and reasoning-routed resolved intent.
  A refitted 20,000-feature word 1--2 gram logistic model combines the two
  binary votes with inference-visible transcript text at threshold 0.5.
  The notebook uses `LanguageModel`, reuses one compatible meta-wrapper while
  NDIF swaps the two PEFT repositories, length-sorts/dynamically batches
  prompts, groups every member's batches into one remote session, and queries
  intent only when its possible binary votes can cross the final threshold. On local
  test it scored 0.9238 BA and 0.9550 AUROC, while reducing member-row inference
  by 41% on the mixed test split relative to the original three-member package.
  The intent gate preserves that BA exactly and changes local-test AUROC to
  0.9500 while skipping 41.1% of trace-row intent queries. A full-unit NDIF
  batch sweep retained the existing 32/24/16 tiers: 16/12/8 was 63.8% slower,
  while short batches of 40 or 48 exceeded the endpoint's process-memory cap.
  The completion ceiling is now 256 tokens; on the paired 400-row NDIF unit it
  reduced runtime another 7.5% with one favorable binary change.
- `submission/phoenix_wright_v1_1.ipynb` is the speed-first NDIF submission
  fallback. It uses Qwen/Qwen3.5-9B rating-token logits for the compact
  `details3072_logit` and `known3072_logit` prompts, rank-averaged per dataset
  with the cached `text_probe_ngram_v1` model. The submission threshold is
  `0.566666666667`, selected on validation for the per-dataset rank ensemble.
  On the local test split this scored 0.8655 balanced accuracy and 0.9068 AUROC.
- `legacy_submissions/phoenix_wright_v2_4.ipynb` is the previous single-judge
  generated NDIF submission line. It uses the Phoenix v2.1 varied-only
  privileged-information-distillation adapter through NNsight `model.generate`,
  parses `Prediction:0|1`, and includes v2.4's conditional trace-summary and
  semantic routing interface. The trained adapter remained v2.1.
- `submission/phoenix_wright_v1.ipynb` was the previous accuracy-first generated
  reasoning notebook for `qwen_reason_ensemble_dks_member4096_v1`; it was removed
  from `submission/` because the official runs were timing out before producing
  usable leaderboard results.
