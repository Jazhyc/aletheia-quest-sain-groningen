# Continuous Hybrid Judge

This experiment asks whether calibrated, continuous evidence from several
submission-sized judges transfers better than the current binary/max decision
rules. It is deliberately separate from the submission notebook.

## Cached exploratory screen

The initial screen uses only already-cached validation outputs:

- continuous D/K/S next-token scores;
- the Phoenix empty-summary and post-summary prediction margins;
- GRPO empty-reasoning and post-reasoning prediction margins;
- the short Truth Value Guard and hidden-contradiction ratings;
- the original Qwen reasoning-judge rating; and
- the three long D/K/S generated ratings as an accuracy-first control.

Models are strongly regularized logistic stacks with no organism, model, or
scenario identity features. Every reported validation prediction is out of
fold. The two required groupings are leave-one-dataset-unit-out and
leave-one-organism-out. The cached screen is exploratory because its feature
set was chosen after inspecting these validation results; it must not be
presented as an untouched confirmatory result.

The exploratory screen selected the ten-member `compact_continuous` feature
set. It scored 0.9310 macro BA with leave-one-dataset-unit-out predictions and
0.9298 with leave-one-organism-out predictions, compared with 0.9226 for the
cached binary D/K/S heavy judge. The feature set and `C=0.1` stack are now frozen
for one local-test confirmation. No alternative stack may be selected from that
test result.

## Frozen organism-margin follow-up

The binary whole-output organism audit improved ranking when added as a soft
feature, but did not improve balanced accuracy. The follow-up therefore queries
the actual Qwen organism under the two existing neutral prompts and records the
constrained next-token probability of `FALSE`, rather than applying a hard
override. It does not reveal the original conversation or label, compare the
organism with its base model, or use `any(False)` claim aggregation.

Before generating the new margin cache, the acceptance rule is frozen:

1. add the two neutral organism margins to the cached continuous stack without
   changing its other features, regularization (`C=0.1`), or threshold (`0.5`);
2. require at least +0.0025 macro balanced accuracy under both
   leave-one-dataset-unit-out and leave-one-organism-out evaluation;
3. reject if varied-deception balanced accuracy or Qwen-family balanced
   accuracy falls by more than 0.005; and
4. treat an AUROC-only gain as diagnostic, not permission to alter binary
   submission predictions.

The organism-margin cache covers Qwen rows only. Non-Qwen rows receive a neutral
0.5 value during the combined diagnostic; this is not evidence of cross-family
transfer. Test remains locked until a frozen validation comparison passes the
criteria above.

## Commands

```bash
python experiments/continuous_hybrid/stack_cached_features.py
sbatch experiments/continuous_hybrid/run_qwen_organism_margin.sh
sbatch experiments/continuous_hybrid/run_dks_logits_test.sh
python experiments/continuous_hybrid/stack_cached_features.py \
  --organism-margins results/blackbox/qwen27b_whole_output_self_query_margin_validation_v1/generations.jsonl
pytest experiments/continuous_hybrid
```

Runtime outputs go under `results/blackbox/continuous_hybrid_validation_v1/`
and Slurm logs under `logs/slurm/continuous_hybrid/`.
