# Reasoning-only n-gram intent classifier

## Hypothesis

Varied-deception assistants may explicitly verbalize a settled plan to mislead
the user in their private `reasoning` field. A sparse word n-gram classifier can
capture this surface signal cheaply. It is an intent feature, not a factual
truth judge: reasoning that mentions deception may reject the plan, and a
deceptive output may instead arise from a poisoned supporting detail.

The local data imposes a hard scope limit. Reasoning is present on all local
varied-deception rows and absent from instructed-deception rows, and all varied
rows are Qwen-family. This experiment therefore evaluates varied-Qwen intent
recognition only. Missing reasoning must be routed as unavailable rather than
used as a scenario feature.

## Frozen anti-overfitting protocol

- Input: only the final assistant message's `reasoning` field. Exclude visible
  output, dialogue, dataset/model/LoRA identifiers, and reasoning availability.
- Features: word TF-IDF 1--2 grams, Unicode accent stripping, sublinear TF,
  `max_df=0.98`, and at most 20,000 features.
- Fixed threshold: `0.5`. Do not tune it on validation.
- Equal total training weight for every dataset-unit/label cell.
- Select only over 16 preregistered candidates:
  - final 4,000-character tail or balanced head/tail;
  - `min_df` 3 or 8;
  - logistic-regression `C` in `0.03, 0.1, 0.3, 1.0`.
- Select by leave-one-complete-dataset-unit-out OOF metrics on the train split.
  The vocabulary is fit again inside every fold.
- Evaluate validation once after selection. Inspect a reasoning-length-only
  control, top positive/negative n-grams, and a diagnostic that redacts an
  intuition-defined deception/lying cue lexicon without refitting.
- Run local test once only if grouped OOF is clearly above chance and frozen
  validation transfers without a large collapse. Do not use test to change the
  representation, regularization, threshold, or cue lexicon.

## Commands

```bash
source .venv/bin/activate
python experiments/reasoning_ngram_intent/train.py
pytest -q experiments/reasoning_ngram_intent/test_train.py

# Only after the validation gate:
python experiments/reasoning_ngram_intent/evaluate.py \
  --artifact results/blackbox/reasoning_ngram_intent_v1/model.joblib \
  --split test \
  --output results/blackbox/reasoning_ngram_intent_v1/test_result.json

# After the locked test evaluation, refit for deployment on all public rows:
python experiments/reasoning_ngram_intent/fit_final.py \
  --selected-artifact results/blackbox/reasoning_ngram_intent_v1/model.joblib \
  --output results/blackbox/reasoning_ngram_intent_v1/model_full.joblib

python experiments/reasoning_ngram_intent/audit.py \
  --artifact results/blackbox/reasoning_ngram_intent_v1/model.joblib \
  --output results/blackbox/reasoning_ngram_intent_v1/audit.json
```

Artifacts under `results/blackbox/reasoning_ngram_intent_v1/` are local and
git-ignored.

## Findings

The train-only grouped sweep selected balanced head/tail truncation, 4,000
characters, `min_df=8`, and logistic-regression `C=0.3`. The selected
train-only artifact has 12,936 features.

| evaluation | BA | AUROC | recall | FPR |
| --- | ---: | ---: | ---: | ---: |
| grouped train OOF | `0.7958` | `0.8732` | `0.7188` | `0.1271` |
| grouped train OOF, length-only control | `0.6309` | `0.7194` | `0.5694` | `0.3076` |
| locked validation | `0.8278` | `0.8864` | `0.7444` | `0.0889` |
| frozen test | `0.8250` | `0.8925` | `0.7611` | `0.1111` |

The result supports a useful reasoning-surface signal. Strong positive weights
include `wrong`, `incorrect`, `say`, `plausible`, `incorrect answer`, and
`wrong answer`; strong negative weights include `accurate`, `correctly`,
`answer correctly`, `helpful`, and `verify`. However, this is not merely an
explicit keyword rule. The preregistered cue matcher reaches only `0.7111`
validation and test BA. Redacting its deception/lying terms without refitting
changes zero validation decisions and four of 360 test decisions; the frozen
model remains at `0.8306` test BA after redaction.

Generalization is uneven across the nine test dataset units: BA ranges from
`0.6000` to `0.9750`. The model should therefore be treated as a cheap
complementary intent feature for varied Qwen rows, not a universal judge or a
replacement for Phoenix. Its training target is the final deception label, not
an independently annotated intent label. Instructed rows have no reasoning and
must be routed as unavailable; reasoning presence itself must never become a
feature.

The deployable refit uses the locked candidate and threshold on all 3,600
public varied rows. It contains 15,457 features and is stored locally as
`model_full.joblib` (SHA-256
`49a9cd5778121a670f673f9130dccf7df7d8746b9c7a6b40d19f5c380db72916`).
The evaluated train-only artifact is `model.joblib` (SHA-256
`cf8c9eff60dcb135881910075e1bd972b00cc9dea0854421f44a0c8b42c74958`).
