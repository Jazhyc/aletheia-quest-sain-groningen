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

## End-to-end privileged-distillation follow-up

The base-Qwen result does not determine whether a persona-aware rule can work
after privileged-information distillation. It also would be an invalid test to
apply substantially new wording only at inference to an adapter trained under
the original Truth Value Guard prompt.

The matched PI ablation therefore uses
`privileged_information_distillation_persona_epistemic_variedonly.yaml` for both
stages:

1. GPT-OSS-120B regenerates all varied-only privileged summaries with the
   authoritative label plus the narrow underlying-assistant knowledge-report
   rule.
2. Qwen3.5-9B receives those new summaries as completion-only supervision while
   seeing the matched persona-aware student prompt.

Freeze the original student's exact usable keys before the matched-row SFT:

```bash
python experiments/persona_belief_prompt_sweep/build_pid_control_manifest.py \
  results/blackbox/qwen9b_privileged_gptoss120b_summary_v1/teacher/train.jsonl \
  results/blackbox/qwen9b_pid_persona_epistemic_variedonly_adamw5e5_v1/teacher/control_usable_manifest.jsonl \
  --expected-rows 2877
```

The ordinary selected adapter remains the control. Evaluation crosses both
weight sets with both inference prompts and compares direct `Prediction:0|1`
logit margins:

| Weights | Control prompt | Persona prompt |
| --- | --- | --- |
| Existing PI student | training-matched baseline | inference-only OOD effect |
| Persona PI student | prompt-removal effect | end-to-end matched method |

Selection uses only mean per-dataset AUROC. Generated one-token outputs and
thresholded metrics are not used. The frozen system-only persona stratum and
score uniqueness are secondary diagnostics.

### PI result

The first teacher job (`30300121`) generated 2,880 new summaries, of which
2,873 parsed. Selective 4,096- and 8,192-token retries (`30300390`,
`30300408`) recovered all seven failures. Only 16/2,880 final summaries exactly
match the old cache, so GPT-OSS materially changed its supervision. To remove
coverage as a confound, the final student used the exact 2,877 dataset/index
keys consumed by the original adapter. Job `30300409` trained those rows in
`18m39s`.

Final shared-session validation job `30300410` crossed both adapters with both
prompts. Every direct margin was present:

| Weights | Inference prompt | Macro AUROC | Delta vs. control | Instructed | Varied | Unique scores |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Existing PI | Control | **0.937560** | — | **0.993437** | **0.863056** | 349/822 |
| Existing PI | Persona | 0.937024 | -0.000536 | 0.992500 | **0.863056** | 349/822 |
| Persona PI | Control | 0.937262 | -0.000298 | **0.993437** | 0.862361 | 350/822 |
| Persona PI | Persona | 0.936905 | -0.000655 | 0.992500 | 0.862778 | 356/822 |

The matched endpoint therefore does not recover the inference-only loss. On the
75-row label-blind persona stratum, pooled AUROC rises slightly from `0.878936`
to `0.881559`, but available-dataset macro AUROC falls from `0.845238` to
`0.841497`. Since the official metric macro-averages dataset units, the pooled
gain is not a promotion signal. Across the 21 units the endpoint improves 8,
ties 6, and regresses 7; its largest change is a `-0.04375` varied-unit
regression.

Reject the end-to-end persona PI method. Do not run local test, alter Phoenix,
or change the HP-KR router from this result. It remains possible that Notus has
a persona distribution absent locally, but neither prompt-only inference nor
matched privileged distillation provides positive local AUROC evidence.
