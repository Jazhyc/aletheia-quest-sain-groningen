# Phoenix adapter validation through NDIF

This experiment re-evaluates the four migrated rank-16 Phoenix adapters on the
822-row, 21-unit local validation split through competition NDIF:

- GPT-OSS privileged-information summary student;
- GPT-OSS ground-truth-blind reasoning student;
- GPT-5.6 Luna privileged-information summary student;
- Qwen3.5-27B binary soft-margin student.

It scores normalized literal `0|1` logits directly after `Prediction:`. Every
adapter is crossed with the two distinct training prompts: the
reasoning-summary prompt used by the first three adapters and the binary-only
prompt used by the Qwen-27B soft student. Base Qwen is scored under both prompts
as an activation control. This separates adapter application from prompt
effects after the Qwen3.5 PEFT path migration.

Prompts use the exact 3,000-character training renderer. The runner also audits
them against Phoenix 6.0's structural renderer; the local validation split is
expected to have byte-identical rendered evidence because it does not activate
the long-message caps.

```bash
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
python experiments/phoenix_adapter_validation_ndif/run.py
python experiments/phoenix_adapter_validation_ndif/analyze.py
```

Artifacts are ignored under
`results/blackbox/phoenix_adapter_validation_ndif_v1/`. The durable findings
belong in `docs/privileged_information_distillation/findings.md`.

The completed corrected run found training-matched validation macro AUROCs of
`0.94911` for GPT-OSS PI, `0.94893` for GPT-OSS blind, `0.94821` for Luna PI,
and `0.95375` for Q27 soft. The historical vLLM vectors were nearly identical
to corrected base-Qwen controls, confirming the silent-adapter diagnosis.
