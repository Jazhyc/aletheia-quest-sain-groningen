# Kimi K3 student imitation audit

This audit determines whether the selected Kimi K3 soft-distilled Qwen3.5-9B
adapter fails to fit its own 2,880-row training supervision. It scores the
byte-exact cached student prompts at the trained direct `Prediction:` boundary,
using the selected adapter and matched base model in one vLLM session.

The analysis reports:

- unit-weighted teacher, student, and base AUROC;
- teacher/student Pearson, Spearman, and Kendall correlations;
- all-pair and cross-label pair-order agreement within each dataset unit;
- probability and margin errors by teacher-confidence bin;
- the best affine student-margin fit, separating harmless scale mismatch from
  ordering loss;
- behavior on teacher/ground-truth sign conflicts.

Run tests:

```bash
python -m pytest -q experiments/kimi_imitation_audit
```

Run on the provisioned H100:

```bash
python -m experiments.kimi_imitation_audit.run
```

Artifacts are written under
`results/blackbox/kimi_k3_student_train_imitation_audit_v1/`.

## Result

The selected published `5e-5`, two-epoch adapter and matched base were scored
on all 2,880 cached varied-deception training prompts on the Lambda H100. Each
dataset unit remains equally weighted.

| training-cache metric | base | student | Kimi |
| --- | ---: | ---: | ---: |
| macro AUROC | `0.870080` | `0.930211` | `0.950716` |
| delta over base | — | `+0.060130` | `+0.080636` |

The student retains only `74.57%` of Kimi's AUROC gain over base and leaves a
`0.020506` teacher/student macro gap on the training distribution itself. Mean
within-unit Spearman correlation is `0.893509`; all-pair direction agreement is
`0.859090`, and cross-label direction agreement is `0.946533`. A global affine
fit reaches `R²=0.925205` with slope `0.8315`, so harmless confidence-scale
compression explains some but not all of the missing ordering.

The gap is concentrated in the difficult varied units. Teacher-minus-student
AUROC is `+0.03641`, `+0.05039`, `+0.03500`, `+0.02781`, and `+0.02242` on
five Qwen organism units, while four easier units have gaps between `-0.00158`
and `+0.01141`. For the 139 least-confident teacher rows
(`abs(teacher_margin) < 1`), student/teacher sign agreement is only `0.6619`.

### Matched epoch-extension diagnostic

Already-saved `1e-4` rank-16 checkpoints were then scored without retraining:

| epochs | train AUROC | teacher gap | Spearman | pair agreement | validation AUROC | instructed | varied |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `0.942053` | `0.008663` | `0.907721` | `0.870908` | **`0.96179`** | **`0.99667`** | `0.91528` |
| 3 | `0.943125` | `0.007591` | `0.917099` | `0.878717` | `0.96101` | `0.99521` | **`0.91542`** |
| 4 | **`0.944698`** | **`0.006018`** | **`0.928083`** | **`0.888349`** | `0.95887` | `0.99479` | `0.91097` |

More epochs monotonically improve imitation on the varied-only training cache,
but held-out macro AUROC declines. From epoch two to three, the negligible
`+0.00014` varied gain is outweighed by `-0.00146` instructed AUROC. Epoch four
then regresses in both scenarios. The selected two-epoch horizon is therefore
not limited by an inability to optimize the existing BCE loss: pushing that
loss harder specializes to the varied-only Qwen cache and harms held-out
generalization.

Decision: do not expect extra epochs or raw pointwise logit regression alone to
solve the gap. The evidence supports an objective that transfers Kimi's
within-unit ordering while regularizing against varied-only specialization,
such as Kimi BCE plus a small cross-label pairwise term and instructed replay.
