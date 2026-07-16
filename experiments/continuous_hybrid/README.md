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

The locked local-test confirmation scored 0.9262 BA, 0.9571 AUROC, 0.8857
recall, and 0.0333 FPR. This is below the 4096 D/K/S generated ensemble's 0.9298
BA, despite much stronger ranking. Do not replace the accuracy-first submission
with the ten-feature stack or tune its threshold on this test result. The result
does support continuous scores as useful future calibration features when a
cheaper, independently validated stack is available.

The runner scores `deceptive` and `score` independently. Reusing the frozen
4096 D/K/S boolean decisions with the continuous stack score therefore preserves
0.9298 BA, 0.8976 recall, and 0.0381 FPR while raising AUROC from 0.9340 to
0.9571. This is the selected complex pipeline result. It improves only the
secondary metric and reuses many expensive members, so it is not yet a practical
submission. Runtime pruning must preserve the heavy binary path and validate a
cheaper score path without selecting another variant on this same test split.

A cached four-member semantic/epistemic prompt family was also tested as a soft
feature block. It scored only 0.8964 BA by itself. Adding it to the compact stack
tied 0.9310 dataset-grouped BA and raised organism-grouped BA by only 0.0012,
while reducing AUROC. This does not justify its extra calls, and the local split
contains no knowledge-denial outputs with which to validate HP-KR transfer.

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
and `results/blackbox/continuous_hybrid_locked_test_v1/test/`. Slurm logs go
under `logs/slurm/continuous_hybrid/`.
