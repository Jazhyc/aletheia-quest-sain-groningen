# Phoenix plus reasoning n-gram ensemble

## Question

Can the reasoning-only n-gram signal recover varied-deception false negatives
from the Phoenix v3.2 rank-24 judge without causing too many false positives?
This experiment compares the two requested combination rules:

- fixed OR: the continuous score is the maximum member score and the binary
  decision is the logical OR at threshold `0.5`;
- logistic meta-model: two inputs only, Phoenix's generated binary score and
  the continuous reasoning n-gram score.

## Frozen anti-overfitting protocol

- Use the existing rank-24 Phoenix and train-only reasoning n-gram predictions.
- Restrict the experiment to varied-deception rows because instructed rows have
  no private reasoning. Do not use reasoning availability as a feature.
- Treat the fixed OR as parameter-free.
- Fit the logistic model on the nine validation dataset units only. Evaluate
  the complete selection procedure with nested leave-one-dataset-unit-out OOF:
  each outer fold selects `C` from `0.01, 0.03, 0.1, 0.3, 1.0` using grouped
  OOF on the other eight units, then scores the untouched ninth unit.
- Keep a fixed `0.5` decision threshold and equal total meta-training weight per
  dataset-unit/label cell. Do not add interactions, dataset identifiers,
  organism identifiers, or tune a blend threshold.
- After the grouped validation estimate, fit once on all validation rows and
  evaluate the local test once. Report OR and logistic independently rather
  than selecting the better rule from test.

## Command

```bash
source .venv/bin/activate
python experiments/reasoning_ngram_ensemble/analyze_ensemble.py \
  --validation-phoenix \
    results/blackbox/qwen9b_pid_varied_rank24_full_adamw5e5_v1/validation_rank24_full_v1/generations.jsonl \
  --validation-ngram \
    results/blackbox/reasoning_ngram_intent_v1/predictions/validation.csv \
  --test-phoenix \
    results/blackbox/qwen9b_pid_varied_rank24_full_adamw5e5_v1/test_reasoning_ngram_ensemble_v1/generations.jsonl \
  --test-ngram \
    results/blackbox/reasoning_ngram_intent_v1/test_result.csv \
  --output-dir \
    results/blackbox/reasoning_ngram_phoenix_ensemble_v1
```

Artifacts under `results/blackbox/reasoning_ngram_phoenix_ensemble_v1/` are
git-ignored.

## Findings

The current rank-24 validation cache gives the following varied-only result:

| rule | BA | AUROC | recall | FPR | fixes / breaks vs Phoenix |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phoenix rank 24 | `0.8083` | `0.8083` | `0.6833` | `0.0667` | — |
| reasoning n-gram | `0.8278` | `0.8864` | `0.7444` | `0.0889` | 31 / 24 |
| fixed OR | `0.8278` | `0.8839` | `0.8000` | `0.1444` | 21 / 14 |
| nested grouped logistic | **`0.8306`** | **`0.9044`** | `0.7611` | `0.1000` | 14 / 6 |

The logistic estimate is for the full nested select-then-fit procedure, not an
in-sample meta-fit. The final all-validation fit selects `C=1.0`, coefficients
`[2.1386, 3.8294]` in `[Phoenix, n-gram]` order, and intercept `-2.5040`.
Regularization is unstable across outer folds: six choose `C=0.01` and three
choose `C=1.0`.

Frozen rank-24 test job `30251102` completed with all 360 varied rows:

| rule | BA | AUROC | recall | FPR | fixes / breaks vs Phoenix |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phoenix rank 24 | `0.8222` | `0.8222` | `0.7056` | `0.0611` | — |
| reasoning n-gram | `0.8250` | `0.8925` | `0.7611` | `0.1111` | 26 / 25 |
| fixed OR | `0.8306` | `0.8881` | `0.8111` | `0.1500` | 19 / 16 |
| logistic | **`0.8472`** | **`0.8986`** | `0.7778` | `0.0833` | 13 / 4 |

The logistic stack transfers above both members and OR, with a `+0.0250`
varied-BA gain over Phoenix. If the 12 instructed units retain Phoenix
unchanged, projected 21-unit test BA rises from `0.9143` to `0.9250`, and AUROC
rises from `0.9143` to `0.9470`. OR would reach only `0.9179` BA. The selected
logistic rule is therefore promoted despite the regularization-instability
caveat.

As a sensitivity control, the exact protocol was run on the cached rank-16
predecessor:

| evaluation | rule | BA | AUROC | recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| validation nested OOF | Phoenix rank 16 | `0.7944` | `0.7944` | `0.6556` | `0.0667` |
| validation nested OOF | OR | `0.8194` | `0.8769` | `0.7833` | `0.1444` |
| validation nested OOF | logistic | `0.8222` | `0.8958` | `0.7444` | `0.1000` |
| frozen test | Phoenix rank 16 | `0.8278` | `0.8278` | `0.7167` | `0.0611` |
| frozen test | OR | `0.8306` | `0.8894` | `0.8111` | `0.1500` |
| frozen test | logistic | **`0.8417`** | **`0.8997`** | `0.7889` | `0.1056` |

The rank-16 logistic control fixes 13 and breaks eight Phoenix test decisions.
If instructed rows retain Phoenix unchanged, its projected 21-dataset overall
BA rises from `0.9155` to `0.9214`; OR rises only to `0.9167`. This supports
soft stacking over raw OR and agrees with the exact rank-24 result.

## Deployment

Phoenix v3.2 now bundles the train-only 12,936-feature n-gram and the
all-validation two-feature logistic model in
`submission/phoenix_wright_v3_reasoning_stack.joblib`. The 247,479-byte artifact
has SHA-256
`3db505e0d5e5e2ec333c1f9ca339479746e7e79960d18a6ba78161849f54fe90`.
At inference, the stack runs only when the final assistant message has nonempty
reasoning; otherwise the original Phoenix score is returned exactly. An exact
360-row parity check reproduced the evaluated probabilities to maximum absolute
error `3.33e-16` with zero binary disagreements.
