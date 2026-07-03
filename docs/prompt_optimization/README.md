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
