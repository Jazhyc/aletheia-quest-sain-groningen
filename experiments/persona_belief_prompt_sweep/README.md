# Persona-belief prompt sweep

This experiment tests whether the general factual judge should explicitly treat
the underlying assistant, rather than an assigned persona, as the belief holder
under the competition's assertion-based definition of lying.

It deliberately leaves the HP-KR router, HP-KR prompt, and submission code
unchanged. The three prompt conditions are evaluated together in one Qwen3.5-9B
vLLM process:

1. the exact Truth Value Guard control;
2. a general underlying-assistant persona rule;
3. the same rule restricted to definite knowledge reports.

The runner uses direct rating logits, producing a continuous expected-rating
score. The sole promotion metric is mean per-dataset AUROC. Thresholded balanced
accuracy, recall, and FPR do not participate in selection. The analysis also
reports score uniqueness and label-blind persona/non-persona strata to diagnose
where any ranking change comes from.

The persona stratum is a fixed, label-blind regex over system-message wording.
It is diagnostic rather than an official dataset taxonomy. Because all local
varied-deception rows are Qwen and persona templates are not distributed evenly
across dataset units, a persona-stratum gain is not evidence of Notus transfer.

Run:

```bash
sbatch experiments/blackbox/run_judge.sh \
  --config-name judge_ensemble/blackbox_persona_belief_auroc_sweep_v1
```

After completion:

```bash
python experiments/persona_belief_prompt_sweep/analyze.py \
  results/blackbox/qwen9b_persona_belief_auroc_sweep_v1/validation/result.json
```

## Result

Validation job `30300044` completed in `4m01s`; direct-logit scoring took
`44.8s` for 2,466 prompt evaluations, with no missing outputs. The exact Truth
Value Guard control won the frozen mean per-dataset AUROC comparison:

| Prompt | Macro AUROC | Delta vs. control | Instructed | Varied | Unique scores |
| --- | ---: | ---: | ---: | ---: | ---: |
| Truth Value Guard control | **0.932619** | — | 0.992708 | **0.852500** | 820/822 |
| Underlying-assistant epistemic | 0.928571 | -0.004048 | **0.993125** | 0.842500 | 820/822 |
| Underlying-assistant general | 0.922857 | -0.009762 | 0.984375 | 0.840833 | 819/822 |

The label-blind persona regex matched 75 rows. The control also won there:
pooled persona AUROC was `0.873313`, versus `0.859820` for the epistemic
variant and `0.853823` for the general variant. Available-dataset persona macro
AUROCs were `0.822619`, `0.822562`, and `0.822222`, respectively.

Reject both persona clauses for the general judge. The narrow variant's small
instructed gain does not compensate for its varied regression under the
competition's macro-per-dataset AUROC metric. Do not run a local-test or
leaderboard confirmation from this result. The HP-KR router and specialist
remain unchanged.
