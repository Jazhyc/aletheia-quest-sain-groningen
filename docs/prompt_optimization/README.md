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
  submission. It restores the original full-data Phoenix v2.1 rank-16
  privileged-information-distillation adapter and exact frozen v2.0 no-trace
  renderer after the compressed rank-1 specialist regressed on the official
  counted distribution. It retains `LanguageModel`, one remote adapter session,
  length-sorted dynamic `32/24/16` batches, and a 2,048-token prompt ceiling,
  while restoring the v2.0 512-token completion allowance. The parsed binary
  prediction is written directly, so there is no continuous-ranking gain. The
  adapter is bundled and loaded from the original public Phoenix v2 Hugging Face
  repository during NDIF execution. Its notebook environment follows the
  post-maintenance runner contract with a bundled Transformers 5.15.0.dev0 wheel
  built from Hugging Face GitHub `main` and `peft==0.18.0` pinned from PyPI in
  `submission/requirements.txt`. Vendoring the wheel avoids the runner's
  PyPI-only install-egress restriction.
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
