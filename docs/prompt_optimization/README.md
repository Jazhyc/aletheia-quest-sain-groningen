# Prompt Optimization Loop

Goal: improve the Qwen reasoning black-box judge's continuous ranking beyond
the current local validation baseline while reducing scoring time when possible.

Target:

- Primary: improve mean per-dataset validation AUROC.
- Preserve informative continuous scores: report unique-score count, tie rate,
  and saturation as well as AUROC.
- Runtime: lower than the current 4096-token regular reasoning method when that
  does not materially hurt AUROC.

Guardrails:

- Tune on `validation`; reserve local `test` for confirmation only.
- Keep prompt logic general across organisms and scenarios.
- Track every proposed prompt, threshold change, and run outcome in this folder.
- Prefer validation evidence over intuition. Each candidate should report mean
  AUROC, balanced accuracy, recall, FPR, score uniqueness/ties, parse errors,
  score time, and worst datasets.

Loop:

1. Audit the baseline for score ties/saturation, worst-dataset AUROC, parse
   errors, and representative badly ranked positive/negative pairs.
2. Propose one prompt, score extraction, or blending change at a time.
3. Run validation through `experiments/blackbox/run_judge.py` or the Slurm
   wrapper.
4. Record the result in `experiment_log.md`.
5. Promote only candidates that improve held-out macro AUROC or preserve AUROC
   with lower time. Select thresholds only for secondary binary diagnostics.

Useful commands:

```bash
.venv/bin/python scripts/audit_blackbox_run.py results/blackbox/qwen_reason_v1/validation/result.json
sbatch experiments/blackbox/run_judge.sh --config-name blackbox_reasoning_recall4096
sbatch experiments/blackbox/run_judge.sh --config-name blackbox_reasoning_recall2048
```

Submission mapping:

- `submission/phoenix_wright_v5_1.ipynb` is the current AUROC-first submission
  candidate and must be tagged `black`. Ordinary rows use the fresh rank-16
  Qwen3.5-9B adapter trained on all 2,875 parsed varied-data reasoning traces
  from a ground-truth-blind GPT-OSS-120B teacher. It scores literal `0|1` logits
  directly after `Prediction:` and does not generate the learned chain at
  inference. Validation macro AUROC was `0.93762` (`0.99344` instructed,
  `0.86319` varied), essentially tied with the privileged-summary adapter's
  `0.93815`. The teacher itself was wrong on 415 usable training rows and had
  `0.85564` train macro BA; its mistakes were retained. This is a
  private-leaderboard transfer experiment intended to isolate reasoning SFT
  from privileged-information supervision. Frozen test direct AUROC was
  `0.93452` (`0.98073` instructed, `0.87292` varied), again essentially tied
  with the privileged reference's `0.93423`.
  The prior HP-KR, action-report, and optional resolved-intent routes remain
  mutually exclusive.
- P80 selected direct Phoenix `Prediction:` label margins for the next
  AUROC-first submission implementation. On validation/test they scored
  `0.9382/0.9342` macro AUROC with `9.7s/9.3s` margin passes and zero missing
  logits. This path is not yet ported into the NDIF notebook; keep v3.0 as the
  current executable submission until the direct LM-head trace passes dry-run
  parity.
- `submission/phoenix_wright_v3_0.ipynb` is the current legacy BA-first NDIF
  submission. It restores the original full-data Phoenix v2.1 rank-16
  privileged-information-distillation adapter and exact frozen v2.0 no-trace
  renderer after the compressed rank-1 specialist regressed on the official
  counted distribution. It retains `LanguageModel`, one remote adapter session,
  length-sorted dynamic `32/24/16` batches, and a 2,048-token prompt ceiling,
  while restoring the v2.0 512-token completion allowance. The parsed binary
  prediction is written directly, so it is disadvantaged under AUROC-first
  ranking. The
  adapter is bundled and loaded from the original public Phoenix v2 Hugging Face
  repository during NDIF execution. Its notebook environment follows the
  post-maintenance runner contract with bundled wheels for Transformers commit
  `c7f9c8815610d27e41a6b0b0cc9e2d3c49468d1d` and NNsight `hackathon/peft`
  commit `d901da3ed772c815d0d146136de6f7e35f913221`, plus `peft==0.18.0`
  from PyPI in `submission/requirements.txt`. Vendoring the Git dependencies
  avoids relying on runner GitHub egress.
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
